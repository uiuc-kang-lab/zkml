#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use group::{Group, Curve};
use ff::Field;
use criterion::{BenchmarkId, Criterion};
use halo2_proofs::{
    halo2curves::pasta::vesta,
    arithmetic::best_multiexp,
};
use rand_core::OsRng;

pub fn bench_ipa_multiexp(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipa_msm");
    
    for k in 15..28 {
        let size = 1 << k;
        let mut rng = OsRng;

        let multiexp_scalars: Vec<vesta::Scalar> = (0..size)
            .map(|_| vesta::Scalar::random(&mut rng))
            .collect();
        let multiexp_bases: Vec<vesta::Affine> = (0..size)
            .map(|_| vesta::Point::random(&mut rng).to_affine())
            .collect();
        
        group.bench_function(BenchmarkId::new("k", k), |b| {
            b.iter(|| {
                best_multiexp(&multiexp_scalars, &multiexp_bases);
            });
        });
    }
}

criterion_group!{
    name=benches; 
    config=Criterion::default().sample_size(10); 
    targets=bench_ipa_multiexp
}
criterion_main!(benches);