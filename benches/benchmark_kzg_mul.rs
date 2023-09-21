#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use ff::Field;
use criterion::Criterion;

use halo2_proofs::halo2curves::bn256::Fr;
use rand_core::OsRng;

pub fn bench_kzg_mul(c: &mut Criterion) {
    let lhs = Fr::random(OsRng);
    let rhs = Fr::random(OsRng);
    c.bench_function("kzg_mul", |b| {
        b.iter(|| criterion::black_box(lhs) * criterion::black_box(rhs));
    });
}

criterion_group!{
    name=benches; 
    config=Criterion::default().sample_size(1000); 
    targets=bench_kzg_mul
}
criterion_main!(benches);