mod nn;

use duckdb::core::LogicalTypeHandle;
use duckdb::ffi::{duckdb_string_t, duckdb_string_t_data, duckdb_string_t_length};
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

fn duckdb_string(word: &duckdb_string_t) -> String {
    unsafe {
        let len = duckdb_string_t_length(*word);
        let c_ptr = duckdb_string_t_data(word as *const _ as *mut _);
        let bytes = std::slice::from_raw_parts(c_ptr as *const u8, len as usize);
        str::from_utf8(bytes).unwrap().to_owned()
    }
}

struct ModelCreate;
impl VScalar for ModelCreate {
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
            .map(duckdb_string)
            .next()
            .unwrap();

        let spec: String = input
            .flat_vector(1)
            .as_slice_with_len::<duckdb_string_t>(input.len())
            .iter()
            .map(duckdb_string)
            .next()
            .unwrap();

        nn::register_model(&modelname, &spec)?;

        //let output_vector = output.flat_vector();
        //output_vector.insert(0, &format!("Created model {modelname}!")[..]);

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
        } else {
            let models = nn::list_models()?;
            let vector = output.flat_vector(0);

            for (idx, model) in models.iter().enumerate() {
                let result = CString::new(model.clone())?;
                vector.insert(idx, result);
            }

            output.set_len(models.len());
        }

        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![])
    }
}

#[duckdb_entrypoint_c_api]
pub unsafe fn extension_entrypoint(con: Connection) -> std::result::Result<(), Box<dyn Error>> {
    con.register_scalar_function::<ModelCreate>("ML_CreateModel")
        .unwrap();
    con.register_table_function::<ListModels>("ML_ListModels")
        .unwrap();

    Ok(())
}
