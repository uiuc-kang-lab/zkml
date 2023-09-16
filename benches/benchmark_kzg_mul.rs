#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use ff::Field;
use criterion::Criterion;

use halo2_proofs::{
    halo2curves::bn256::G1Affine, halo2curves::CurveAffine,
};
use rand_core::OsRng;

pub fn bench_kzg_mul(c: &mut Criterion) {
    c.bench_function("kzg_mul", |b| {
        let lhs = G1Affine::random(OsRng);
        let rhs = <G1Affine as CurveAffine>::ScalarExt::random(OsRng);
        b.iter(|| criterion::black_box(lhs) * criterion::black_box(rhs));
    });
}

criterion_group!{
    name=benches; 
    config=Criterion::default().sample_size(100); 
    targets=bench_kzg_mul
}
criterion_main!(benches);