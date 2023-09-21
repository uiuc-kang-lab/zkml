#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use ff::Field;
use criterion::Criterion;

use halo2_proofs::halo2curves::pasta::vesta;
use rand_core::OsRng;

pub fn bench_ipa_mul(c: &mut Criterion) {
    let mut rng = OsRng;
    let lhs = vesta::Scalar::random(&mut rng);
    let rhs = vesta::Scalar::random(&mut rng);
    c.bench_function("ipa_mul", |b| {    
        b.iter(|| criterion::black_box(lhs) * criterion::black_box(rhs));
    });
}

criterion_group!{
    name=benches;
    config=Criterion::default().sample_size(1000); 
    targets=bench_ipa_mul
}
criterion_main!(benches);