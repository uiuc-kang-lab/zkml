#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use ff::Field;
use criterion::{BenchmarkId, Criterion};
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::{
    poly::ipa::commitment::ParamsIPA,
    halo2curves::pasta::vesta,
    arithmetic::best_multiexp,
};
use rand_core::OsRng;

pub fn bench_ipa_multiexp(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipa_msm");
    
    for k in 13..20 {
        let size = 1 << k;
        let mut rng = OsRng;

        let multiexp_scalars = (0..size)
            .map(|_| vesta::Scalar::random(&mut rng))
            .collect::<Vec<_>>();
        let params = ParamsIPA::<vesta::Affine>::new(k);
        let multiexp_bases = params.get_g();
        
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