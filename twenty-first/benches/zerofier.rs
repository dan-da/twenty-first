use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BatchSize;
use criterion::Criterion;
use rand::thread_rng;
use rand::Rng;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::traits::PrimitiveRootOfUnity;

criterion_main!(benches);
criterion_group!(
    benches,
    zerofier<0>,
    zerofier<10>,
    zerofier<100>,
    zerofier<1_000>,
    zerofier<10_000>,
    fast_zerofier<0>,
    fast_zerofier<10>,
    fast_zerofier<100>,
    fast_zerofier<1_000>,
    fast_zerofier<10_000>,
);

fn zerofier<const N: usize>(c: &mut Criterion) {
    let mut rng = thread_rng();
    let id = format!("zerofier {N}");
    c.bench_function(&id, |b| {
        b.iter_batched(
            || rng.gen(),
            |v: [BFieldElement; N]| Polynomial::zerofier(&v),
            BatchSize::SmallInput,
        )
    });
}

fn fast_zerofier<const N: usize>(c: &mut Criterion) {
    let mut rng = thread_rng();
    let root_order = (N + 1).next_power_of_two();
    let root_of_unity = BFieldElement::primitive_root_of_unity(root_order as u64).unwrap();

    let id = format!("fast_zerofier {N}");
    c.bench_function(&id, |b| {
        b.iter_batched(
            || rng.gen(),
            |v: [BFieldElement; N]| Polynomial::fast_zerofier(&v, root_of_unity, root_order),
            BatchSize::SmallInput,
        )
    });
}
