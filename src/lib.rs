mod nn;
mod utils;

use nn::TrainingConfig;

use duckdb::core::LogicalTypeHandle;
use duckdb::ffi::duckdb_string_t;
use duckdb::vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab};
use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeId},
    vscalar::{ScalarFunctionSignature, VScalar},
    vtab::arrow::WritableVector,
    Connection,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use libduckdb_sys as ffi;
use std::error::Error;
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

use utils::{duckdb_list_to_vec_f32, duckdb_string_to_string, write_vec_to_output};

struct CreateModel;
impl VScalar for CreateModel {
    type State = ();

    unsafe fn invoke(
        _state: &(),
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let modelname: String = input
            .flat_vector(0)
            .as_slice_with_len::<duckdb_string_t>(input.len())
            .iter()
            .map(duckdb_string_to_string)
            .next()
            .unwrap();

        let spec: String = input
            .flat_vector(1)
            .as_slice_with_len::<duckdb_string_t>(input.len())
            .iter()
            .map(duckdb_string_to_string)
            .next()
            .unwrap();

        nn::register_model(&modelname, &spec)?;

        let output_vector = output.flat_vector();
        output_vector.insert(0, "Ok");

        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Varchar.into(), LogicalTypeId::Varchar.into()],
            LogicalTypeId::Varchar.into(),
        )]
    }
}

struct ListModels;
impl VTab for ListModels {
    type InitData = AtomicBool;
    type BindData = ();

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("model", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("json", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        Ok(())
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(AtomicBool::new(false))
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        if func.get_init_data().swap(true, Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }

        let models = nn::list_models()?;
        output.set_len(models.len());

        for (idx, (model, json)) in models.into_iter().enumerate() {
            output.flat_vector(0).insert(idx, CString::new(model)?);
            output.flat_vector(1).insert(idx, CString::new(json)?);
        }

        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![])
    }
}

struct TrainModel;
impl VScalar for TrainModel {
    type State = ();

    unsafe fn invoke(
        _state: &(),
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let modelname: String = input
            .flat_vector(0)
            .as_slice_with_len::<duckdb_string_t>(1)
            .iter()
            .map(duckdb_string_to_string)
            .next()
            .unwrap();

        let model = nn::get_model(&modelname)?;

        let features = duckdb_list_to_vec_f32(input.list_vector(1), input.len());
        let targets = duckdb_list_to_vec_f32(input.list_vector(2), input.len());

        let mut training_config = TrainingConfig::new();

        if input.num_columns() > 3 {
            training_config = serde_json::from_str(
                &input
                    .flat_vector(3)
                    .as_slice_with_len::<duckdb_string_t>(1)
                    .iter()
                    .map(duckdb_string_to_string)
                    .next()
                    .unwrap(),
            )?;
        }
        println!("Training {modelname}: Hyperparameters {training_config:?}");

        let flatvec = output.flat_vector();

        // Important! Write empty string to outputs to prevent crashes on scalar func calls
        for (idx, _) in features.iter().enumerate() {
            flatvec.insert(idx, "");
        }

        let model = nn::train_model_reg(model, features.clone(), targets, &training_config);

        nn::put_model(&modelname, model)?;

        let targets = nn::predict(&modelname, features)?;
        write_vec_to_output(targets, output);

        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![
            ScalarFunctionSignature::exact(
                vec![
                    LogicalTypeId::Varchar.into(),
                    LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
                    LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
                ],
                LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
            ),
            ScalarFunctionSignature::exact(
                vec![
                    LogicalTypeId::Varchar.into(),
                    LogicalTypeHandle::list(&LogicalTypeId::Float.into()),
                    LogicalTypeHandle::list(&LogicalTypeId::Float.into()),
                    LogicalTypeId::Varchar.into(),
                    //LogicalTypeHandle::struct_type(&[
                    //    ("epochs", LogicalTypeId::Integer.into()),
                    //    ("batchsize", LogicalTypeId::Integer.into()),
                    //    ("learningrate", LogicalTypeId::Float.into()),
                    //]),
                ],
                LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
            ),
        ]
    }
}

struct ModelPredict;
impl VScalar for ModelPredict {
    type State = ();

    unsafe fn invoke(
        _state: &(),
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let modelname: String = input
            .flat_vector(0)
            .as_slice_with_len::<duckdb_string_t>(input.len())
            .iter()
            .map(duckdb_string_to_string)
            .next()
            .unwrap();

        let features = duckdb_list_to_vec_f32(input.list_vector(1), input.len());
        let targets = nn::predict(&modelname, features)?;

        write_vec_to_output(targets, output);

        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeId::Varchar.into(),
                LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
            ],
            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
        )]
    }
}

#[duckdb_entrypoint_c_api]
pub unsafe fn extension_entrypoint(con: Connection) -> std::result::Result<(), Box<dyn Error>> {
    con.register_scalar_function::<CreateModel>("ml_create")
        .unwrap();
    con.register_table_function::<ListModels>("ml_list")
        .unwrap();
    con.register_scalar_function::<TrainModel>("ml_train")
        .unwrap();
    con.register_scalar_function::<ModelPredict>("ml_pred")
        .unwrap();

    Ok(())
}
