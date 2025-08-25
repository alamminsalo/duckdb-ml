use duckdb::core::ListVector;
use duckdb::ffi::*;
use duckdb::vtab::arrow::WritableVector;

pub fn duckdb_string_to_string(word: &duckdb_string_t) -> String {
    unsafe {
        let len = duckdb_string_t_length(*word);
        let c_ptr = duckdb_string_t_data(word as *const _ as *mut _);
        let bytes = std::slice::from_raw_parts(c_ptr as *const u8, len as usize);
        str::from_utf8(bytes).unwrap().to_owned()
    }
}

pub fn duckdb_list_to_vec_f32(list: ListVector, rows: usize) -> Vec<Vec<f32>> {
    list.child(list.len())
        .as_slice()
        .iter()
        .copied()
        .collect::<Vec<f32>>()
        .chunks(list.len() / rows)
        .map(|c| c.to_vec())
        .collect()
}

pub fn write_vec_to_output(data: Vec<Vec<f32>>, output: &mut dyn WritableVector) {
    let mut values = Vec::new();
    let mut offsets = vec![0usize];
    let value_len = data.get(0).map(|v| v.len()).unwrap_or_default();

    for inner_vec in data {
        values.extend(inner_vec);
        offsets.push(values.len());
    }

    let vec = &mut output.list_vector();
    vec.set_child(&values);

    for (idx, offset) in offsets.into_iter().enumerate() {
        vec.set_entry(idx, offset, value_len);
    }
}
