use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime::RescuePrime;
use twenty_first::shared_math::rescue_prime_params as params;
use twenty_first::shared_math::traits::GetPrimitiveRootOfUnity;

// Benchmark the RescuePrime AIR constraints compilation.

fn rescue_prime_air_constraints(criterion: &mut Criterion) {
    // let mut rp_bench = rescue_prime_params_bfield_0();
    let mut rp_bench: RescuePrime = params::rescue_prime_small_test_params();
    rp_bench.round_count = 1;
    rp_bench.alpha = 4;
    let omicron = BFieldElement::ring_zero()
        .get_primitive_root_of_unity(1 << 5)
        .0
        .unwrap();

    let mut group = criterion.benchmark_group("rescue_prime_air_constraints");
    group.sample_size(10);
    let benchmark_id = BenchmarkId::from_parameter(omicron);
    group.bench_with_input(benchmark_id, &omicron, |bencher, _omicron| {
        bencher.iter(|| rp_bench.get_air_constraints(omicron));
    });
}

criterion_group!(benches, rescue_prime_air_constraints);
criterion_main!(benches);