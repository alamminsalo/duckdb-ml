mod nn;
mod utils;

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

use utils::{duckdb_list_to_vec_f32, duckdb_string_to_string};

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
            .as_slice_with_len::<duckdb_string_t>(input.len())
            .iter()
            .map(duckdb_string_to_string)
            .next()
            .unwrap();

        let model = nn::get_model(&modelname)?;

        let features = duckdb_list_to_vec_f32(input.list_vector(1), input.len());
        let targets = duckdb_list_to_vec_f32(input.list_vector(2), input.len());

        println!("Train model={model} inputs={features:?} outputs={targets:?}");

        let model = nn::train_model_reg(model, features, targets);

        nn::put_model(&modelname, model)?;

        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeId::Varchar.into(),
                LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
                LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
            ],
            LogicalTypeId::Varchar.into(),
        )]
    }
}

// struct TrainModelData {
//     model: String,
//     features: Vec<Vec<f32>>,
//     targets: Vec<Vec<f32>>,
// }
//
// impl VTab for TrainModel {
//     type InitData = AtomicBool;
//     type BindData = TrainModelData;
//
//     fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
//         bind.add_result_column("model", LogicalTypeHandle::from(LogicalTypeId::Varchar));
//         bind.add_result_column("epoch", LogicalTypeHandle::from(LogicalTypeId::Varchar));
//         bind.add_result_column("train loss", LogicalTypeHandle::from(LogicalTypeId::Float));
//         bind.add_result_column("test loss", LogicalTypeHandle::from(LogicalTypeId::Float));
//
//         let model = bind.get_parameter(0).to_string();
//         let features = duckdb_list_to_vec_f32(bind.get_parameter(1).try_into().unwrap()), 0);
//
//         //let features = bind.get_parameter(1);
//         //let targets = bind.get_parameter(2);
//
//         //println!("{features:?}, {targets:?}");
//
//         Ok(TrainModelData {
//             model,
//             //features,
//             //targets,
//         })
//     }
//
//     fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
//         Ok(AtomicBool::new(false))
//     }
//
//     fn func(
//         func: &TableFunctionInfo<Self>,
//         output: &mut DataChunkHandle,
//     ) -> Result<(), Box<dyn Error>> {
//         if func.get_init_data().swap(true, Ordering::Relaxed) {
//             output.set_len(0);
//             return Ok(());
//         }
//
//         Ok(())
//     }
//
//     fn parameters() -> Option<Vec<LogicalTypeHandle>> {
//         Some(vec![
//             LogicalTypeHandle::from(LogicalTypeId::Varchar),
//             LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
//             LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
//         ])
//     }
// }

#[duckdb_entrypoint_c_api]
pub unsafe fn extension_entrypoint(con: Connection) -> std::result::Result<(), Box<dyn Error>> {
    con.register_scalar_function::<CreateModel>("ML_CreateModel")
        .unwrap();
    con.register_table_function::<ListModels>("ML_ListModels")
        .unwrap();
    con.register_scalar_function::<TrainModel>("ML_TrainModel")
        .unwrap();

    Ok(())
}
