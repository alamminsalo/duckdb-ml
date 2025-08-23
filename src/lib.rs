mod nn;

use duckdb::ffi::{duckdb_string_t, duckdb_string_t_data, duckdb_string_t_length};
use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeId},
    vscalar::{ScalarFunctionSignature, VScalar},
    vtab::arrow::WritableVector,
    Connection,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use libduckdb_sys as ffi;
use std::error::Error;
use std::sync::atomic::AtomicBool;

#[repr(C)]
struct HelloBindData {
    name: String,
}

#[repr(C)]
struct HelloInitData {
    done: AtomicBool,
}

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
        let model: Vec<String> = input
            .flat_vector(0)
            .as_slice_with_len::<duckdb_string_t>(input.len())
            .iter()
            .map(duckdb_string)
            .collect();

        let specfile: Vec<String> = input
            .flat_vector(1)
            .as_slice_with_len::<duckdb_string_t>(input.len())
            .iter()
            .map(duckdb_string)
            .collect();

        for (idx, (model, specfile)) in model.iter().zip(specfile).enumerate() {
            (model, specfile);

            let output_vector = output.flat_vector();
            output_vector.insert(idx, &model[..]);
        }

        Ok(())
    }

    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeId::Varchar.into(), LogicalTypeId::Varchar.into()],
            LogicalTypeId::Varchar.into(),
        )]
    }
}

//impl VTab for HelloVTab {
//    type InitData = HelloInitData;
//    type BindData = HelloBindData;
//
//    fn bind(bind: &BindInfo) -> Result<Self::BindData> {
//        bind.add_result_column("column0", LogicalTypeHandle::from(LogicalTypeId::Varchar));
//        let name = bind.get_parameter(0).to_string();
//        Ok(HelloBindData { name })
//    }
//
//    fn init(_: &InitInfo) -> Result<Self::InitData> {
//        Ok(HelloInitData {
//            done: AtomicBool::new(false),
//        })
//    }
//
//    fn func(func: &TableFunctionInfo<Self>, output: &mut DataChunkHandle) -> Result<()> {
//        let init_data = func.get_init_data();
//        let bind_data = func.get_bind_data();
//        if init_data.done.swap(true, Ordering::Relaxed) {
//            output.set_len(0);
//        } else {
//            let vector = output.flat_vector(0);
//            let result = CString::new(format!("Rusty Quack {} 🐥", bind_data.name))?;
//            vector.insert(0, result);
//            output.set_len(1);
//        }
//        Ok(())
//    }
//
//    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
//        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
//    }
//}

#[duckdb_entrypoint_c_api]
pub unsafe fn extension_entrypoint(
    con: Connection,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    //LazyCell::force(&SERIALIZER);

    con.register_scalar_function::<ModelCreate>("model_create")
        .expect("Failed to register hg_encode() function");

    Ok(())
}
