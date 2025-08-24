use duckdb::core::ListVector;
use duckdb::ffi::*;

pub fn duckdb_string_to_string(word: &duckdb_string_t) -> String {
    unsafe {
        let len = duckdb_string_t_length(*word);
        let c_ptr = duckdb_string_t_data(word as *const _ as *mut _);
        let bytes = std::slice::from_raw_parts(c_ptr as *const u8, len as usize);
        str::from_utf8(bytes).unwrap().to_owned()
    }
}

pub fn duckdb_list_to_vec_f32(list: ListVector, rows: usize) -> Vec<Vec<f32>> {
    let array_size = list.len() / rows;
    println!("array len {array_size}");

    list.child(list.len())
        .as_slice()
        .iter()
        .copied()
        .collect::<Vec<f32>>()
        .chunks(array_size)
        .map(|c| c.to_vec())
        .collect()
}
