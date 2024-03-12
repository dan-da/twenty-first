#[inline]
pub(super) fn serialize<B>(b: &B) -> Vec<u8>
where
    B: ?Sized + serde::Serialize,
{
    bincode::serialize(b).expect("should have serialized value")
}

#[inline]
pub(super) fn deserialize<'b, B>(bytes: &'b [u8]) -> B
where
    B: serde::de::Deserialize<'b>,
{
    bincode::deserialize(bytes).expect("should have deserialized value")
}
