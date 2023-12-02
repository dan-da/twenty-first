use itertools::Itertools;
use rusty_leveldb::{WriteBatch, DB};
use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    sync::{Arc, Mutex, MutexGuard},
};

use super::storage_vec::{Index, StorageVec};
use crate::shared_math::b_field_element::BFieldElement;

pub enum WriteOperation<ParentKey, ParentValue> {
    Write(ParentKey, ParentValue),
    Delete(ParentKey),
}

pub trait DbTable<ParentKey, ParentValue> {
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>>;
    fn restore_or_new(&mut self);
}

pub trait StorageReader<ParentKey, ParentValue> {
    /// Return multiple values from storage, in the same order as the input keys
    fn get_many(&mut self, keys: &[ParentKey]) -> Vec<Option<ParentValue>>;

    /// Return a single value from storage
    fn get(&mut self, key: ParentKey) -> Option<ParentValue>;
}

pub enum VecWriteOperation<Index, T> {
    OverWrite((Index, T)),
    Push(T),
    Pop,
}

pub struct DbtVec<ParentKey, ParentValue, Index, T> {
    reader: Arc<Mutex<dyn StorageReader<ParentKey, ParentValue> + Send + Sync>>,
    current_length: Option<Index>,
    key_prefix: u8,
    write_queue: VecDeque<VecWriteOperation<Index, T>>,
    cache: HashMap<Index, T>,
    name: String,
}

impl<ParentKey, ParentValue, Index, T> DbtVec<ParentKey, ParentValue, Index, T>
where
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    ParentKey: From<Index>,
    Index: From<ParentValue> + From<u64> + Clone,
{
    // Return the key of ParentKey type used to store the length of the vector
    fn get_length_key(key_prefix: u8) -> ParentKey {
        let const_length_key: ParentKey = 0u8.into();
        let key_prefix_key: ParentKey = key_prefix.into();
        (key_prefix_key, const_length_key).into()
    }

    /// Return the length at the last write to disk
    fn persisted_length(&self) -> Option<Index> {
        self.reader
            .lock()
            .expect("Could not get lock on DbtVec object (persisted_length).")
            .get(Self::get_length_key(self.key_prefix))
            .map(|v| v.into())
    }

    /// Return the key of ParentKey type used to store the element at a given index of Index type
    fn get_index_key(&self, index: Index) -> ParentKey {
        let key_prefix_key: ParentKey = self.key_prefix.into();
        let index_key: ParentKey = index.into();
        (key_prefix_key, index_key).into()
    }

    pub fn new(
        reader: Arc<Mutex<dyn StorageReader<ParentKey, ParentValue> + Send + Sync>>,
        key_prefix: u8,
        name: &str,
    ) -> Self {
        let length = None;
        let cache = HashMap::new();
        Self {
            key_prefix,
            reader,
            write_queue: VecDeque::default(),
            current_length: length,
            cache,
            name: name.to_string(),
        }
    }
}

/// A trait that does things with already acquired locks
trait WithLock<ParentKey, ParentValue, T> {
    fn len_with_lock(&self, lock: &MutexGuard<DbtVec<ParentKey, ParentValue, Index, T>>) -> Index;
    fn write_op_overwrite_with_lock(
        &self,
        lock: &mut MutexGuard<DbtVec<ParentKey, ParentValue, Index, T>>,
        index: Index,
        value: T,
    );
}

impl<ParentKey, ParentValue, T> WithLock<ParentKey, ParentValue, T>
    for Arc<Mutex<DbtVec<ParentKey, ParentValue, Index, T>>>
where
    ParentKey: From<Index>,
    ParentValue: From<T>,
    T: Clone + From<ParentValue> + Debug,
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    Index: From<ParentValue> + From<u64>,
{
    fn len_with_lock(&self, lock: &MutexGuard<DbtVec<ParentKey, ParentValue, u64, T>>) -> Index {
        lock.current_length
            .unwrap_or_else(|| lock.persisted_length().unwrap_or(0))
    }

    fn write_op_overwrite_with_lock(
        &self,
        lock: &mut MutexGuard<DbtVec<ParentKey, ParentValue, Index, T>>,
        index: Index,
        value: T,
    ) {
        if let Some(_old_val) = lock.cache.insert(index, value.clone()) {
            // If cache entry exists, we remove any corresponding
            // OverWrite ops in the `write_queue` to reduce disk IO.

            // logic: retain all ops that are not overwrite, and
            // overwrite ops that do not have an index matching cache_index.
            lock.write_queue.retain(|op| match op {
                VecWriteOperation::OverWrite((i, _)) => *i != index,
                _ => true,
            })
        }

        lock.write_queue
            .push_back(VecWriteOperation::OverWrite((index, value)));
    }
}

impl<ParentKey, ParentValue, T> StorageVec<T>
    for Arc<Mutex<DbtVec<ParentKey, ParentValue, Index, T>>>
where
    ParentKey: From<Index>,
    ParentValue: From<T>,
    T: Clone + From<ParentValue> + Debug,
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    Index: From<ParentValue> + From<u64>,
{
    /// note: acquires lock
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// note: acquires lock
    fn len(&self) -> Index {
        self.len_with_lock(
            &self
                .lock()
                .expect("Could not get lock on DbtVec as StorageVec"),
        )
    }

    fn get(&self, index: Index) -> T {
        // Disallow getting values out-of-bounds
        let lock = self
            .lock()
            .expect("Could not get lock on DbtVec as StorageVec");

        assert!(
            index < self.len_with_lock(&lock),
            "Out-of-bounds. Got {index} but length was {}. persisted vector name: {}",
            self.len_with_lock(&lock),
            lock.name
        );

        // try cache first
        if lock.cache.contains_key(&index) {
            return lock.cache.get(&index).unwrap().clone();
        }

        // then try persistent storage
        let key: ParentKey = lock.get_index_key(index);
        let val = lock
            .reader
            .lock()
            .expect("Could not get lock on DbtVec object")
            .get(key)
            .unwrap_or_else(|| {
                panic!(
                    "Element with index {index} does not exist in {}. This should not happen",
                    lock.name
                )
            });
        val.into()
    }

    /// Fetch multiple elements from a `DbtVec` and return the elements matching the order
    /// of the input indices.
    fn get_many(&self, indices: &[Index]) -> Vec<T> {
        fn sort_to_match_requested_index_order<T>(indexed_elements: HashMap<usize, T>) -> Vec<T> {
            let mut elements = indexed_elements.into_iter().collect_vec();
            elements.sort_unstable_by_key(|&(index_position, _)| index_position);
            elements.into_iter().map(|(_, element)| element).collect()
        }

        let self_lock = self
            .lock()
            .expect("Could not get lock on DbtVec as StorageVec");

        let self_length = self.len_with_lock(&self_lock);

        // Do *not* refer to `self` after this point, instead use `self_lock`
        assert!(
            indices.iter().all(|x| *x < self_length),
            "Out-of-bounds. Got indices {indices:?} but length was {}. persisted vector name: {}",
            self_length,
            self_lock.name
        );

        let (indices_of_elements_in_cache, indices_of_elements_not_in_cache): (Vec<_>, Vec<_>) =
            indices
                .iter()
                .copied()
                .enumerate()
                .partition(|&(_, index)| self_lock.cache.contains_key(&index));

        let mut fetched_elements = HashMap::with_capacity(indices.len());
        for (index_position, index) in indices_of_elements_in_cache {
            let value = self_lock.cache.get(&index).unwrap().clone();
            fetched_elements.insert(index_position, value);
        }

        let no_need_to_lock_database = indices_of_elements_not_in_cache.is_empty();
        if no_need_to_lock_database {
            return sort_to_match_requested_index_order(fetched_elements);
        }

        let mut db_reader = self_lock
            .reader
            .lock()
            .expect("Could not get lock on StorageReader object (get_many 3).");

        let keys_for_indices_not_in_cache = indices_of_elements_not_in_cache
            .iter()
            .map(|&(_, index)| self_lock.get_index_key(index))
            .collect_vec();
        let elements_fetched_from_db = db_reader
            .get_many(&keys_for_indices_not_in_cache)
            .into_iter()
            .map(|x| x.unwrap().into());

        let indexed_fetched_elements_from_db = indices_of_elements_not_in_cache
            .iter()
            .map(|&(index_position, _)| index_position)
            .zip_eq(elements_fetched_from_db);
        fetched_elements.extend(indexed_fetched_elements_from_db);

        sort_to_match_requested_index_order(fetched_elements)
    }

    /// Return all stored elements in a vector, whose index matches the StorageVec's.
    /// It's the caller's responsibility that there is enough memory to store all elements.
    fn get_all(&self) -> Vec<T> {
        let self_lock = self
            .lock()
            .expect("Could not get lock on DbtVec as StorageVec");

        let length = self.len_with_lock(&self_lock);

        let (indices_of_elements_in_cache, indices_of_elements_not_in_cache): (Vec<_>, Vec<_>) =
            (0..length).partition(|index| self_lock.cache.contains_key(index));

        let mut fetched_elements: Vec<Option<T>> = vec![None; length as usize];
        for index in indices_of_elements_in_cache {
            let element = self_lock.cache[&index].clone();
            fetched_elements[index as usize] = Some(element);
        }

        let no_need_to_lock_database = indices_of_elements_not_in_cache.is_empty();
        if no_need_to_lock_database {
            return fetched_elements
                .into_iter()
                .map(|x| x.unwrap())
                .collect_vec();
        }

        let keys = indices_of_elements_not_in_cache
            .iter()
            .map(|x| self_lock.get_index_key(*x))
            .collect_vec();
        let mut db_reader = self_lock
            .reader
            .lock()
            .expect("Could not get lock on StorageReader object");
        let elements_fetched_from_db = db_reader
            .get_many(&keys)
            .into_iter()
            .map(|x| x.unwrap().into());
        let indexed_fetched_elements_from_db = indices_of_elements_not_in_cache
            .into_iter()
            .zip_eq(elements_fetched_from_db);
        for (index, element) in indexed_fetched_elements_from_db {
            fetched_elements[index as usize] = Some(element);
        }

        fetched_elements
            .into_iter()
            .map(|x| x.unwrap())
            .collect_vec()
    }

    fn set(&mut self, index: Index, value: T) {
        // Disallow setting values out-of-bounds
        let mut self_lock = self
            .lock()
            .expect("Could not get lock on DbtVec as StorageVec");

        let self_len = self.len_with_lock(&self_lock);

        assert!(
            index < self_len,
            "Out-of-bounds. Got {index} but length was {}. persisted vector name: {}",
            self_len,
            self_lock.name
        );

        self.write_op_overwrite_with_lock(&mut self_lock, index, value);
    }

    /// set multiple elements.
    ///
    /// panics if key_vals contains an index not in the collection
    ///
    /// It is the caller's responsibility to ensure that index values are
    /// unique.  If not, the last value with the same index will win.
    /// For unordered collections such as HashMap, the behavior is undefined.
    fn set_many(&mut self, key_vals: impl IntoIterator<Item = (Index, T)>) {
        let mut self_lock = self
            .lock()
            .expect("Could not get lock on DbtVec as StorageVec");

        let self_len = self.len_with_lock(&self_lock);

        for (index, value) in key_vals.into_iter() {
            assert!(
                index < self_len,
                "Out-of-bounds. Got {index} but length was {}. persisted vector name: {}",
                self_len,
                self_lock.name
            );

            self.write_op_overwrite_with_lock(&mut self_lock, index, value);
        }
    }

    fn pop(&mut self) -> Option<T> {
        let mut self_lock = self
            .lock()
            .expect("Could not get lock on DbtVec as StorageVec");

        // If vector is empty, return None
        if self.len_with_lock(&self_lock) == 0 {
            return None;
        }

        // add to write queue
        self_lock.write_queue.push_back(VecWriteOperation::Pop);

        // Update length
        *self_lock.current_length.as_mut().unwrap() -= 1;

        // try cache first
        let current_length = self.len_with_lock(&self_lock);
        if self_lock.cache.contains_key(&current_length) {
            self_lock.cache.remove(&current_length)
        } else {
            // then try persistent storage
            let key = self_lock.get_index_key(current_length);
            self_lock
                .reader
                .lock()
                .expect("Could not get lock on DbtVec object.")
                .get(key)
                .map(|value| value.into())
        }
    }

    fn push(&mut self, value: T) {
        let mut self_lock = self
            .lock()
            .expect("Could not get lock on DbtVec as StorageVec)");

        // add to write queue
        self_lock
            .write_queue
            .push_back(VecWriteOperation::Push(value.clone()));

        // record in cache
        let current_length = self.len_with_lock(&self_lock);
        let _old_val = self_lock.cache.insert(current_length, value);

        // note: we cannot naively remove any previous `Push` ops with
        // this value from the write_queue (to reduce disk i/o) because
        // there might be corresponding `Pop` op(s).

        // update length
        self_lock.current_length = Some(current_length + 1);
    }
}

impl<ParentKey, ParentValue, T> DbTable<ParentKey, ParentValue>
    for DbtVec<ParentKey, ParentValue, Index, T>
where
    ParentKey: From<Index>,
    ParentValue: From<T>,
    T: Clone,
    T: From<ParentValue>,
    ParentKey: From<(ParentKey, ParentKey)>,
    ParentKey: From<u8>,
    Index: From<ParentValue>,
    ParentValue: From<Index>,
{
    /// Collect all added elements that have not yet been persisted
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>> {
        let maybe_original_length = self.persisted_length();
        // necessary because we need maybe_original_length.is_none() later
        let original_length = maybe_original_length.unwrap_or(0);
        let mut length = original_length;
        let mut queue = vec![];
        while let Some(write_element) = self.write_queue.pop_front() {
            match write_element {
                VecWriteOperation::OverWrite((i, t)) => {
                    let key = self.get_index_key(i);
                    queue.push(WriteOperation::Write(key, Into::<ParentValue>::into(t)));
                }
                VecWriteOperation::Push(t) => {
                    let key = self.get_index_key(length);
                    length += 1;
                    queue.push(WriteOperation::Write(key, Into::<ParentValue>::into(t)));
                }
                VecWriteOperation::Pop => {
                    let key = self.get_index_key(length - 1);
                    length -= 1;
                    queue.push(WriteOperation::Delete(key));
                }
            };
        }

        if original_length != length || maybe_original_length.is_none() {
            let key = Self::get_length_key(self.key_prefix);
            queue.push(WriteOperation::Write(
                key,
                Into::<ParentValue>::into(length),
            ));
        }

        self.cache.clear();

        queue
    }

    fn restore_or_new(&mut self) {
        if let Some(length) = self
            .reader
            .lock()
            .expect("Could not get lock on DbtVec object (restore_or_new).")
            .get(Self::get_length_key(self.key_prefix))
        {
            self.current_length = Some(length.into());
        } else {
            self.current_length = Some(0);
        }
        self.cache.clear();
        self.write_queue.clear();
    }
}

// possible future extension
// pub struct DbtHashMap<Key, Value, K, V> {
//     parent: Arc<Mutex<DbtSchema<Key, Value>>>,
// }

pub trait StorageSingleton<T>
where
    T: Clone,
{
    fn get(&self) -> T;
    fn set(&mut self, t: T);
}

pub struct DbtSingleton<ParentKey, ParentValue, T> {
    current_value: T,
    old_value: T,
    key: ParentKey,
    reader: Arc<Mutex<dyn StorageReader<ParentKey, ParentValue> + Sync + Send>>,
}

impl<ParentKey, ParentValue, T> StorageSingleton<T>
    for Arc<Mutex<DbtSingleton<ParentKey, ParentValue, T>>>
where
    T: Clone + From<ParentValue>,
{
    fn get(&self) -> T {
        self.lock()
            .expect("Could not get lock on DbtSingleton object (get).")
            .current_value
            .clone()
    }

    fn set(&mut self, t: T) {
        self.lock()
            .expect("Could not get lock on DbtSingleton object (set).")
            .current_value = t;
    }
}

impl<ParentKey, ParentValue, T> DbTable<ParentKey, ParentValue>
    for DbtSingleton<ParentKey, ParentValue, T>
where
    T: Eq + Clone + Default + From<ParentValue>,
    ParentValue: From<T> + Debug,
    ParentKey: Clone,
{
    fn pull_queue(&mut self) -> Vec<WriteOperation<ParentKey, ParentValue>> {
        if self.current_value == self.old_value {
            vec![]
        } else {
            self.old_value = self.current_value.clone();
            vec![WriteOperation::Write(
                self.key.clone(),
                self.current_value.clone().into(),
            )]
        }
    }

    fn restore_or_new(&mut self) {
        self.current_value = match self
            .reader
            .lock()
            .expect("Could not get lock on DbTable object (restore_or_new).")
            .get(self.key.clone())
        {
            Some(value) => value.into(),
            None => T::default(),
        }
    }
}

pub struct DbtSchema<ParentKey, ParentValue, Reader: StorageReader<ParentKey, ParentValue>> {
    pub tables: Vec<Arc<Mutex<dyn DbTable<ParentKey, ParentValue> + Send + Sync>>>,
    pub reader: Arc<Mutex<Reader>>,
}

impl<
        ParentKey,
        ParentValue,
        Reader: StorageReader<ParentKey, ParentValue> + 'static + Sync + Send,
    > DbtSchema<ParentKey, ParentValue, Reader>
{
    pub fn new_vec<Index, T>(
        &mut self,
        name: &str,
    ) -> Arc<Mutex<DbtVec<ParentKey, ParentValue, Index, T>>>
    where
        ParentKey: From<Index> + 'static,
        ParentValue: From<T> + 'static,
        T: Clone + From<ParentValue> + 'static,
        ParentKey: From<(ParentKey, ParentKey)>,
        ParentKey: From<u8>,
        Index: From<ParentValue>,
        ParentValue: From<Index>,
        Index: From<u64> + 'static,
        DbtVec<ParentKey, ParentValue, Index, T>: DbTable<ParentKey, ParentValue> + Send + Sync,
    {
        assert!(self.tables.len() < 255);
        let reader = self.reader.clone();
        let vector = DbtVec::<ParentKey, ParentValue, Index, T> {
            reader,
            current_length: None,
            key_prefix: self.tables.len() as u8,
            write_queue: VecDeque::new(),
            cache: HashMap::new(),
            name: name.to_string(),
        };
        let arc_mutex_vector = Arc::new(Mutex::new(vector));
        self.tables.push(arc_mutex_vector.clone());
        arc_mutex_vector
    }

    // possible future extension
    // fn new_hashmap<K, V>(&self) -> Arc<Mutex<DbtHashMap<K, V>>> { }

    pub fn new_singleton<S>(
        &mut self,
        key: ParentKey,
    ) -> Arc<Mutex<DbtSingleton<ParentKey, ParentValue, S>>>
    where
        S: Default + Eq + Clone + 'static,
        ParentKey: 'static,
        ParentValue: From<S> + 'static,
        ParentKey: From<(ParentKey, ParentKey)> + From<u8>,
        DbtSingleton<ParentKey, ParentValue, S>: DbTable<ParentKey, ParentValue> + Send + Sync,
    {
        let reader = self.reader.clone();
        let singleton = DbtSingleton::<ParentKey, ParentValue, S> {
            current_value: S::default(),
            old_value: S::default(),
            key,
            reader,
        };
        let arc_mutex_singleton = Arc::new(Mutex::new(singleton));
        self.tables.push(arc_mutex_singleton.clone());
        arc_mutex_singleton
    }
}

pub trait StorageWriter<ParentKey, ParentValue> {
    fn persist(&mut self);
    fn restore_or_new(&mut self);
}

#[derive(Clone, PartialEq, Eq)]
pub struct RustyKey(pub Vec<u8>);
impl From<u8> for RustyKey {
    fn from(value: u8) -> Self {
        Self([value].to_vec())
    }
}
impl From<(RustyKey, RustyKey)> for RustyKey {
    fn from(value: (RustyKey, RustyKey)) -> Self {
        let v0 = value.0 .0;
        let v1 = value.1 .0;
        RustyKey([v0, v1].concat())
    }
}
impl From<u64> for RustyKey {
    fn from(value: u64) -> Self {
        RustyKey(value.to_be_bytes().to_vec())
    }
}

#[derive(Debug)]
pub struct RustyValue(pub Vec<u8>);

impl From<RustyValue> for u64 {
    fn from(value: RustyValue) -> Self {
        u64::from_be_bytes(value.0.try_into().unwrap())
    }
}
impl From<u64> for RustyValue {
    fn from(value: u64) -> Self {
        RustyValue(value.to_be_bytes().to_vec())
    }
}
impl From<RustyValue> for crate::shared_math::tip5::Digest {
    fn from(value: RustyValue) -> Self {
        crate::shared_math::tip5::Digest::new(
            value
                .0
                .chunks(8)
                .map(|ch| {
                    u64::from_be_bytes(ch.try_into().expect("Cannot cast RustyValue into Digest"))
                })
                .map(BFieldElement::new)
                .collect::<Vec<_>>()
                .try_into().expect("Can cast RustyValue into BFieldElements but number does not match that of Digest."),
        )
    }
}
impl From<crate::shared_math::tip5::Digest> for RustyValue {
    fn from(value: crate::shared_math::tip5::Digest) -> Self {
        RustyValue(
            value
                .values()
                .map(|b| b.value())
                .map(u64::to_be_bytes)
                .concat(),
        )
    }
}

pub struct RustyReader {
    pub db: Arc<Mutex<DB>>,
}

impl StorageReader<RustyKey, RustyValue> for RustyReader {
    fn get(&mut self, key: RustyKey) -> Option<RustyValue> {
        self.db
            .lock()
            .expect("StorageReader for RustyWallet: could not get database lock for reading (get)")
            .get(&key.0)
            .map(RustyValue)
    }

    fn get_many(&mut self, keys: &[RustyKey]) -> Vec<Option<RustyValue>> {
        let mut lock = self.db.lock().expect("Could not get lock");

        let mut res = vec![];
        for key in keys {
            res.push(lock.get(&key.0).map(RustyValue));
        }

        res
    }
}

/// Database schema and tables logic for RustyLevelDB. You probably
/// want to implement your own storage class after this example so
/// that you can hardcode the schema in new(). But it is nevertheless
/// possible to use this struct and add to the scheme after calling
/// new() (that's what the tests do).
pub struct SimpleRustyStorage {
    db: Arc<Mutex<DB>>,
    schema: DbtSchema<RustyKey, RustyValue, SimpleRustyReader>,
}

impl StorageWriter<RustyKey, RustyValue> for SimpleRustyStorage {
    fn persist(&mut self) {
        let mut write_batch = WriteBatch::new();
        for table in &self.schema.tables {
            let operations = table
                .lock()
                .expect("Could not get lock on table object in SimpleRustyStorage::persist.")
                .pull_queue();
            for op in operations {
                match op {
                    WriteOperation::Write(key, value) => write_batch.put(&key.0, &value.0),
                    WriteOperation::Delete(key) => write_batch.delete(&key.0),
                }
            }
        }

        self.db
            .lock()
            .expect("Persist: could not get lock on database.")
            .write(write_batch, true)
            .expect("Could not persist to database.");
    }

    fn restore_or_new(&mut self) {
        for table in &self.schema.tables {
            table
                .lock()
                .expect("Could not get lock on table obect in SimpleRustyStorage::restore_or_new.")
                .restore_or_new();
        }
    }
}

impl SimpleRustyStorage {
    pub fn new(db: DB) -> Self {
        let db_pointer = Arc::new(Mutex::new(db));
        let reader = SimpleRustyReader {
            db: db_pointer.clone(),
        };
        let schema = DbtSchema::<RustyKey, RustyValue, SimpleRustyReader> {
            tables: Vec::new(),
            reader: Arc::new(Mutex::new(reader)),
        };
        Self {
            db: db_pointer,
            schema,
        }
    }

    pub fn close(&mut self) {
        self.db
            .lock()
            .expect("SimpleRustyStorage::close: could not get lock on database.")
            .close()
            .expect("Could not close database.");
    }
}

struct SimpleRustyReader {
    db: Arc<Mutex<DB>>,
}

impl StorageReader<RustyKey, RustyValue> for SimpleRustyReader {
    fn get(&mut self, key: RustyKey) -> Option<RustyValue> {
        self.db
            .lock()
            .expect("get: could not get lock on database (for reading)")
            .get(&key.0)
            .map(RustyValue)
    }

    fn get_many(&mut self, keys: &[RustyKey]) -> Vec<Option<RustyValue>> {
        let mut db_lock = self
            .db
            .lock()
            .expect("get_many: could not get lock on database (for reading)");
        keys.iter()
            .map(|key| db_lock.get(&key.0).map(RustyValue))
            .collect()
    }
}

#[cfg(test)]
mod tests {

    use rand::{random, Rng, RngCore};

    use crate::shared_math::other::random_elements;

    use super::*;

    #[derive(Default, PartialEq, Eq, Clone, Debug)]
    struct S(Vec<u8>);
    impl From<Vec<u8>> for S {
        fn from(value: Vec<u8>) -> Self {
            S(value)
        }
    }
    impl From<S> for Vec<u8> {
        fn from(value: S) -> Self {
            value.0
        }
    }
    impl From<(S, S)> for S {
        fn from(value: (S, S)) -> Self {
            let vector0: Vec<u8> = value.0.into();
            let vector1: Vec<u8> = value.1.into();
            S([vector0, vector1].concat())
        }
    }
    impl From<RustyValue> for S {
        fn from(value: RustyValue) -> Self {
            Self(value.0)
        }
    }
    impl From<S> for RustyValue {
        fn from(value: S) -> Self {
            Self(value.0)
        }
    }
    impl From<S> for u64 {
        fn from(value: S) -> Self {
            u64::from_be_bytes(value.0.try_into().unwrap())
        }
    }

    #[test]
    fn test_simple_singleton() {
        let singleton_value = S([1u8, 3u8, 3u8, 7u8].to_vec());
        // let opt = rusty_leveldb::Options::default();
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        let mut rusty_storage = SimpleRustyStorage::new(db);
        let mut singleton = rusty_storage
            .schema
            .new_singleton::<S>(RustyKey([1u8; 1].to_vec()));

        // initialize
        rusty_storage.restore_or_new();

        // test
        assert_eq!(singleton.get(), S([].to_vec()));

        // set
        singleton.set(singleton_value.clone());

        // test
        assert_eq!(singleton.get(), singleton_value);

        // persist
        rusty_storage.persist();

        // test
        assert_eq!(singleton.get(), singleton_value);

        // drop
        rusty_storage.close();

        // restore
        let new_db = DB::open("test-database", opt).unwrap();
        let mut new_rusty_storage = SimpleRustyStorage::new(new_db);
        let new_singleton = new_rusty_storage
            .schema
            .new_singleton::<S>(RustyKey([1u8; 1].to_vec()));
        new_rusty_storage.restore_or_new();

        // test
        assert_eq!(new_singleton.get(), singleton_value);
    }

    #[test]
    fn test_simple_vector() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        let mut rusty_storage = SimpleRustyStorage::new(db);
        let mut vector = rusty_storage.schema.new_vec::<u64, S>("test-vector");

        // initialize
        rusty_storage.restore_or_new();

        // should work to pass empty array, when vector.is_empty() == true
        vector.set_all(&[]);

        // test `get_all`
        assert!(
            vector.get_all().is_empty(),
            "`get_all` on unpopulated vector must return empty vector"
        );

        // populate
        vector.push(S([1u8].to_vec()));
        vector.push(S([3u8].to_vec()));
        vector.push(S([4u8].to_vec()));
        vector.push(S([7u8].to_vec()));
        vector.push(S([8u8].to_vec()));

        // test `get`
        assert_eq!(vector.get(0), S([1u8].to_vec()));
        assert_eq!(vector.get(1), S([3u8].to_vec()));
        assert_eq!(vector.get(2), S([4u8].to_vec()));
        assert_eq!(vector.get(3), S([7u8].to_vec()));
        assert_eq!(vector.get(4), S([8u8].to_vec()));
        assert_eq!(vector.len(), 5);

        // test `get_many`
        assert_eq!(
            vector.get_many(&[0, 2, 3]),
            vec![vector.get(0), vector.get(2), vector.get(3)]
        );
        assert_eq!(
            vector.get_many(&[2, 3, 0]),
            vec![vector.get(2), vector.get(3), vector.get(0)]
        );
        assert_eq!(
            vector.get_many(&[3, 0, 2]),
            vec![vector.get(3), vector.get(0), vector.get(2)]
        );
        assert_eq!(
            vector.get_many(&[0, 1, 2, 3, 4]),
            vec![
                vector.get(0),
                vector.get(1),
                vector.get(2),
                vector.get(3),
                vector.get(4),
            ]
        );
        assert_eq!(vector.get_many(&[]), vec![]);
        assert_eq!(vector.get_many(&[3]), vec![vector.get(3)]);

        // We allow `get_many` to take repeated indices.
        assert_eq!(vector.get_many(&[3; 0]), vec![vector.get(3); 0]);
        assert_eq!(vector.get_many(&[3; 1]), vec![vector.get(3); 1]);
        assert_eq!(vector.get_many(&[3; 2]), vec![vector.get(3); 2]);
        assert_eq!(vector.get_many(&[3; 3]), vec![vector.get(3); 3]);
        assert_eq!(vector.get_many(&[3; 4]), vec![vector.get(3); 4]);
        assert_eq!(vector.get_many(&[3; 5]), vec![vector.get(3); 5]);
        assert_eq!(
            vector.get_many(&[3, 3, 2, 3]),
            vec![vector.get(3), vector.get(3), vector.get(2), vector.get(3)]
        );

        // at this point, `vector` should contain:
        let expect_values = vec![
            S([1u8].to_vec()),
            S([3u8].to_vec()),
            S([4u8].to_vec()),
            S([7u8].to_vec()),
            S([8u8].to_vec()),
        ];

        // test `get_all`
        assert_eq!(
            expect_values,
            vector.get_all(),
            "`get_all` must return expected values"
        );

        // test roundtrip through `set_all`, `get_all`
        let values_tmp = vec![
            S([2u8].to_vec()),
            S([4u8].to_vec()),
            S([6u8].to_vec()),
            S([8u8].to_vec()),
            S([9u8].to_vec()),
        ];
        vector.set_all(&values_tmp);

        assert_eq!(
            values_tmp,
            vector.get_all(),
            "`get_all` must return values passed to `set_all`",
        );

        vector.set_all(&expect_values);

        // persist
        rusty_storage.persist();

        // test `get_all` after persist
        assert_eq!(
            expect_values,
            vector.get_all(),
            "`get_all` must return expected values after persist"
        );

        // modify
        let last = vector.pop().unwrap();

        // test
        assert_eq!(last, S([8u8].to_vec()));

        // drop without persisting
        rusty_storage.close();

        // create new database
        let new_db = DB::open("test-database", opt).unwrap();
        let mut new_rusty_storage = SimpleRustyStorage::new(new_db);
        let mut new_vector = new_rusty_storage.schema.new_vec::<u64, S>("test-vector");

        // initialize
        new_rusty_storage.restore_or_new();

        // modify
        new_vector.set(2, S([3u8].to_vec()));
        let last_again = new_vector.pop().unwrap();
        assert_eq!(last_again, S([8u8].to_vec()));

        // test
        assert_eq!(new_vector.get(0), S([1u8].to_vec()));
        assert_eq!(new_vector.get(1), S([3u8].to_vec()));
        assert_eq!(new_vector.get(2), S([3u8].to_vec()));
        assert_eq!(new_vector.get(3), S([7u8].to_vec()));
        assert_eq!(new_vector.len(), 4);

        // test `get_many`, ensure that output matches input ordering
        assert_eq!(new_vector.get_many(&[2]), vec![new_vector.get(2)]);
        assert_eq!(
            new_vector.get_many(&[3, 1, 0]),
            vec![new_vector.get(3), new_vector.get(1), new_vector.get(0)]
        );
        assert_eq!(
            new_vector.get_many(&[0, 2, 3]),
            vec![new_vector.get(0), new_vector.get(2), new_vector.get(3)]
        );
        assert_eq!(
            new_vector.get_many(&[0, 1, 2, 3]),
            vec![
                new_vector.get(0),
                new_vector.get(1),
                new_vector.get(2),
                new_vector.get(3),
            ]
        );
        assert_eq!(new_vector.get_many(&[]), vec![]);
        assert_eq!(new_vector.get_many(&[3]), vec![new_vector.get(3)]);

        // We allow `get_many` to take repeated indices.
        assert_eq!(new_vector.get_many(&[3; 0]), vec![new_vector.get(3); 0]);
        assert_eq!(new_vector.get_many(&[3; 1]), vec![new_vector.get(3); 1]);
        assert_eq!(new_vector.get_many(&[3; 2]), vec![new_vector.get(3); 2]);
        assert_eq!(new_vector.get_many(&[3; 3]), vec![new_vector.get(3); 3]);
        assert_eq!(new_vector.get_many(&[3; 4]), vec![new_vector.get(3); 4]);
        assert_eq!(new_vector.get_many(&[3; 5]), vec![new_vector.get(3); 5]);

        // test `get_all`
        assert_eq!(
            vec![
                S([1u8].to_vec()),
                S([3u8].to_vec()),
                S([3u8].to_vec()),
                S([7u8].to_vec()),
            ],
            new_vector.get_all(),
            "`get_all` must return expected values"
        );

        new_vector.set(1, S([130u8].to_vec()));
        assert_eq!(
            vec![
                S([1u8].to_vec()),
                S([130u8].to_vec()),
                S([3u8].to_vec()),
                S([7u8].to_vec()),
            ],
            new_vector.get_all(),
            "`get_all` must return expected values, after mutation"
        );
    }

    #[test]
    fn test_dbtcvecs_get_many() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        let mut rusty_storage = SimpleRustyStorage::new(db);
        let mut vector = rusty_storage.schema.new_vec::<u64, S>("test-vector");

        // initialize
        rusty_storage.restore_or_new();

        // populate
        const TEST_LIST_LENGTH: u8 = 105;
        for i in 0u8..TEST_LIST_LENGTH {
            vector.push(S(vec![i, i, i]));
        }

        let read_indices: Vec<u64> = random_elements::<u64>(30)
            .into_iter()
            .map(|x| x % TEST_LIST_LENGTH as u64)
            .collect();
        let values = vector.get_many(&read_indices);
        assert!(read_indices
            .iter()
            .zip(values)
            .all(|(index, value)| value == S(vec![*index as u8, *index as u8, *index as u8])));

        // Mutate some indices
        let mutate_indices: Vec<u64> = random_elements::<u64>(30)
            .into_iter()
            .map(|x| x % TEST_LIST_LENGTH as u64)
            .collect();
        for index in mutate_indices.iter() {
            vector.set(
                *index,
                S(vec![*index as u8 + 1, *index as u8 + 1, *index as u8 + 1]),
            )
        }

        let new_values = vector.get_many(&read_indices);
        for (value, index) in new_values.into_iter().zip(read_indices) {
            if mutate_indices.contains(&index) {
                assert_eq!(
                    S(vec![index as u8 + 1, index as u8 + 1, index as u8 + 1]),
                    value
                )
            } else {
                assert_eq!(S(vec![index as u8, index as u8, index as u8]), value)
            }
        }
    }

    #[test]
    fn test_dbtcvecs_set_many_get_many() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        // initialize storage
        let mut rusty_storage = SimpleRustyStorage::new(db);
        rusty_storage.restore_or_new();
        let mut vector = rusty_storage.schema.new_vec::<u64, S>("test-vector");

        // Generate initial index/value pairs.
        const TEST_LIST_LENGTH: u8 = 105;
        let init_keyvals: Vec<(Index, S)> = (0u8..TEST_LIST_LENGTH)
            .map(|i| (i as Index, S(vec![i, i, i])))
            .collect();

        // set_many() does not grow the list, so we must first push
        // some empty elems, to desired length.
        for _ in 0u8..TEST_LIST_LENGTH {
            vector.push(S(vec![]));
        }

        // set the initial values
        vector.set_many(init_keyvals);

        // generate some random indices to read
        let read_indices: Vec<u64> = random_elements::<u64>(30)
            .into_iter()
            .map(|x| x % TEST_LIST_LENGTH as u64)
            .collect();

        // perform read, and validate as expected
        let values = vector.get_many(&read_indices);
        assert!(read_indices
            .iter()
            .zip(values)
            .all(|(index, value)| value == S(vec![*index as u8, *index as u8, *index as u8])));

        // Generate some random indices for mutation
        let mutate_indices: Vec<u64> = random_elements::<u64>(30)
            .iter()
            .map(|x| x % TEST_LIST_LENGTH as u64)
            .collect();

        // Generate keyvals for mutation
        let mutate_keyvals: Vec<(Index, S)> = mutate_indices
            .iter()
            .map(|index| {
                let val = (index % TEST_LIST_LENGTH as u64 + 1) as u8;
                (*index, S(vec![val, val, val]))
            })
            .collect();

        // Mutate values at randomly generated indices
        vector.set_many(mutate_keyvals);

        // Verify mutated values, and non-mutated also.
        let new_values = vector.get_many(&read_indices);
        for (value, index) in new_values.into_iter().zip(read_indices.clone()) {
            if mutate_indices.contains(&index) {
                assert_eq!(
                    S(vec![index as u8 + 1, index as u8 + 1, index as u8 + 1]),
                    value
                )
            } else {
                assert_eq!(S(vec![index as u8, index as u8, index as u8]), value)
            }
        }

        // Persist and verify that result is unchanged
        rusty_storage.persist();
        let new_values_after_persist = vector.get_many(&read_indices);
        for (value, index) in new_values_after_persist.into_iter().zip(read_indices) {
            if mutate_indices.contains(&index) {
                assert_eq!(
                    S(vec![index as u8 + 1, index as u8 + 1, index as u8 + 1]),
                    value
                )
            } else {
                assert_eq!(S(vec![index as u8, index as u8, index as u8]), value)
            }
        }
    }

    #[test]
    fn test_dbtcvecs_set_all_get_many() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        // initialize storage
        let mut rusty_storage = SimpleRustyStorage::new(db);
        rusty_storage.restore_or_new();
        let mut vector = rusty_storage.schema.new_vec::<u64, S>("test-vector");

        // Generate initial index/value pairs.
        const TEST_LIST_LENGTH: u8 = 105;
        let init_vals: Vec<S> = (0u8..TEST_LIST_LENGTH)
            .map(|i| (S(vec![i, i, i])))
            .collect();

        let mut mutate_vals = init_vals.clone(); // for later

        // set_all() does not grow the list, so we must first push
        // some empty elems, to desired length.
        for _ in 0u8..TEST_LIST_LENGTH {
            vector.push(S(vec![]));
        }

        // set the initial values
        vector.set_all(&init_vals);

        // generate some random indices to read
        let read_indices: Vec<u64> = random_elements::<u64>(30)
            .into_iter()
            .map(|x| x % TEST_LIST_LENGTH as u64)
            .collect();

        // perform read, and validate as expected
        let values = vector.get_many(&read_indices);
        assert!(read_indices
            .iter()
            .zip(values)
            .all(|(index, value)| value == S(vec![*index as u8, *index as u8, *index as u8])));

        // Generate some random indices for mutation
        let mutate_indices: Vec<u64> = random_elements::<u64>(30)
            .iter()
            .map(|x| x % TEST_LIST_LENGTH as u64)
            .collect();

        // Generate vals for mutation
        for index in mutate_indices.iter() {
            let val = (index % TEST_LIST_LENGTH as u64 + 1) as u8;
            mutate_vals[*index as usize] = S(vec![val, val, val]);
        }

        // Mutate values at randomly generated indices
        vector.set_all(&mutate_vals);

        // Verify mutated values, and non-mutated also.
        let new_values = vector.get_many(&read_indices);
        for (value, index) in new_values.into_iter().zip(read_indices) {
            if mutate_indices.contains(&index) {
                assert_eq!(
                    S(vec![index as u8 + 1, index as u8 + 1, index as u8 + 1]),
                    value
                )
            } else {
                assert_eq!(S(vec![index as u8, index as u8, index as u8]), value)
            }
        }
    }

    #[test]
    fn storage_schema_vector_pbt() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        let mut rusty_storage = SimpleRustyStorage::new(db);
        let mut persisted_vector = rusty_storage.schema.new_vec::<u64, u64>("test-vector");

        // Insert 1000 elements
        let mut rng = rand::thread_rng();
        let mut normal_vector = vec![];
        for _ in 0..1000 {
            let value = random();
            normal_vector.push(value);
            persisted_vector.push(value);
        }
        rusty_storage.persist();

        for _ in 0..1000 {
            match rng.gen_range(0..=5) {
                0 => {
                    // `push`
                    let push_val = rng.next_u64();
                    persisted_vector.push(push_val);
                    normal_vector.push(push_val);
                }
                1 => {
                    // `pop`
                    let persisted_pop_val = persisted_vector.pop().unwrap();
                    let normal_pop_val = normal_vector.pop().unwrap();
                    assert_eq!(persisted_pop_val, normal_pop_val);
                }
                2 => {
                    // `get_many`
                    assert_eq!(normal_vector.len(), persisted_vector.len() as usize);

                    let index = rng.gen_range(0..normal_vector.len());
                    assert_eq!(Vec::<u64>::default(), persisted_vector.get_many(&[]));
                    assert_eq!(normal_vector[index], persisted_vector.get(index as u64));
                    assert_eq!(
                        vec![normal_vector[index]],
                        persisted_vector.get_many(&[index as u64])
                    );
                    assert_eq!(
                        vec![normal_vector[index], normal_vector[index]],
                        persisted_vector.get_many(&[index as u64, index as u64])
                    );
                }
                3 => {
                    // `set`
                    let value = rng.next_u64();
                    let index = rng.gen_range(0..normal_vector.len());
                    normal_vector[index] = value;
                    persisted_vector.set(index as u64, value);
                }
                4 => {
                    // `set_many`
                    let indices: Vec<u64> = (0..rng.gen_range(0..10))
                        .map(|_| rng.gen_range(0..normal_vector.len() as u64))
                        .unique()
                        .collect();
                    let values: Vec<u64> = (0..indices.len()).map(|_| rng.next_u64()).collect_vec();
                    let update: Vec<(u64, u64)> =
                        indices.into_iter().zip_eq(values.into_iter()).collect();
                    for (key, val) in update.iter() {
                        normal_vector[*key as usize] = *val;
                    }
                    persisted_vector.set_many(update);
                }
                5 => {
                    // persist
                    rusty_storage.persist();
                }
                _ => unreachable!(),
            }
        }

        // Check equality after above loop
        assert_eq!(normal_vector.len(), persisted_vector.len() as usize);
        for (i, nvi) in normal_vector.iter().enumerate() {
            assert_eq!(*nvi, persisted_vector.get(i as u64));
        }

        // Check equality using `get_many`
        assert_eq!(
            normal_vector,
            persisted_vector.get_many(&(0..normal_vector.len() as u64).collect_vec())
        );

        // Check equality after persisting updates
        rusty_storage.persist();
        assert_eq!(normal_vector.len(), persisted_vector.len() as usize);
        for (i, nvi) in normal_vector.iter().enumerate() {
            assert_eq!(*nvi, persisted_vector.get(i as u64));
        }

        // Check equality using `get_many`
        assert_eq!(
            normal_vector,
            persisted_vector.get_many(&(0..normal_vector.len() as u64).collect_vec())
        );
    }

    #[test]
    fn test_two_vectors_and_singleton() {
        let singleton_value = S([3u8, 3u8, 3u8, 1u8].to_vec());
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        let mut rusty_storage = SimpleRustyStorage::new(db);
        let mut vector1 = rusty_storage.schema.new_vec::<u64, S>("test-vector1");
        let mut vector2 = rusty_storage.schema.new_vec::<u64, S>("test-vector2");
        let mut singleton = rusty_storage
            .schema
            .new_singleton::<S>(RustyKey([1u8; 1].to_vec()));

        // initialize
        rusty_storage.restore_or_new();

        assert!(
            vector1.get_all().is_empty(),
            "`get_all` call to unpopulated persistent vector must return empty vector"
        );
        assert!(
            vector2.get_all().is_empty(),
            "`get_all` call to unpopulated persistent vector must return empty vector"
        );

        // populate 1
        vector1.push(S([1u8].to_vec()));
        vector1.push(S([30u8].to_vec()));
        vector1.push(S([4u8].to_vec()));
        vector1.push(S([7u8].to_vec()));
        vector1.push(S([8u8].to_vec()));

        // populate 2
        vector2.push(S([1u8].to_vec()));
        vector2.push(S([3u8].to_vec()));
        vector2.push(S([3u8].to_vec()));
        vector2.push(S([7u8].to_vec()));

        // set singleton
        singleton.set(singleton_value.clone());

        // modify 1
        vector1.set(0, S([8u8].to_vec()));

        // test
        assert_eq!(vector1.get(0), S([8u8].to_vec()));
        assert_eq!(vector1.get(1), S([30u8].to_vec()));
        assert_eq!(vector1.get(2), S([4u8].to_vec()));
        assert_eq!(vector1.get(3), S([7u8].to_vec()));
        assert_eq!(vector1.get(4), S([8u8].to_vec()));
        assert_eq!(
            vector1.get_many(&[2, 0, 3]),
            vec![vector1.get(2), vector1.get(0), vector1.get(3)]
        );
        assert_eq!(
            vector1.get_many(&[2, 3, 1]),
            vec![vector1.get(2), vector1.get(3), vector1.get(1)]
        );
        assert_eq!(vector1.len(), 5);
        assert_eq!(vector2.get(0), S([1u8].to_vec()));
        assert_eq!(vector2.get(1), S([3u8].to_vec()));
        assert_eq!(vector2.get(2), S([3u8].to_vec()));
        assert_eq!(vector2.get(3), S([7u8].to_vec()));
        assert_eq!(
            vector2.get_many(&[0, 1, 2]),
            vec![vector2.get(0), vector2.get(1), vector2.get(2)]
        );
        assert_eq!(vector2.get_many(&[]), vec![]);
        assert_eq!(
            vector2.get_many(&[1, 2]),
            vec![vector2.get(1), vector2.get(2)]
        );
        assert_eq!(
            vector2.get_many(&[2, 1]),
            vec![vector2.get(2), vector2.get(1)]
        );
        assert_eq!(vector2.len(), 4);
        assert_eq!(singleton.get(), singleton_value);
        assert_eq!(
            vec![
                S([8u8].to_vec()),
                S([30u8].to_vec()),
                S([4u8].to_vec()),
                S([7u8].to_vec()),
                S([8u8].to_vec())
            ],
            vector1.get_all()
        );
        assert_eq!(
            vec![
                S([1u8].to_vec()),
                S([3u8].to_vec()),
                S([3u8].to_vec()),
                S([7u8].to_vec()),
            ],
            vector2.get_all()
        );

        // persist and drop
        rusty_storage.persist();
        assert_eq!(
            vector2.get_many(&[2, 1]),
            vec![vector2.get(2), vector2.get(1)]
        );
        rusty_storage.close();

        // restore from disk
        let new_db = DB::open("test-database", opt).unwrap();
        let mut new_rusty_storage = SimpleRustyStorage::new(new_db);
        let new_vector1 = new_rusty_storage.schema.new_vec::<u64, S>("test-vector1");
        let mut new_vector2 = new_rusty_storage.schema.new_vec::<u64, S>("test-vector2");
        new_rusty_storage.restore_or_new();

        // test again
        assert_eq!(new_vector1.get(0), S([8u8].to_vec()));
        assert_eq!(new_vector1.get(1), S([30u8].to_vec()));
        assert_eq!(new_vector1.get(2), S([4u8].to_vec()));
        assert_eq!(new_vector1.get(3), S([7u8].to_vec()));
        assert_eq!(new_vector1.get(4), S([8u8].to_vec()));
        assert_eq!(new_vector1.len(), 5);
        assert_eq!(new_vector2.get(0), S([1u8].to_vec()));
        assert_eq!(new_vector2.get(1), S([3u8].to_vec()));
        assert_eq!(new_vector2.get(2), S([3u8].to_vec()));
        assert_eq!(new_vector2.get(3), S([7u8].to_vec()));
        assert_eq!(new_vector2.len(), 4);
        assert_eq!(singleton.get(), singleton_value);

        // Test `get_many` for a restored DB
        assert_eq!(
            new_vector2.get_many(&[2, 1]),
            vec![new_vector2.get(2), new_vector2.get(1)]
        );
        assert_eq!(
            new_vector2.get_many(&[0, 1]),
            vec![new_vector2.get(0), new_vector2.get(1)]
        );
        assert_eq!(
            new_vector2.get_many(&[1, 0]),
            vec![new_vector2.get(1), new_vector2.get(0)]
        );
        assert_eq!(
            new_vector2.get_many(&[0, 1, 2, 3]),
            vec![
                new_vector2.get(0),
                new_vector2.get(1),
                new_vector2.get(2),
                new_vector2.get(3),
            ]
        );
        assert_eq!(new_vector2.get_many(&[2]), vec![new_vector2.get(2),]);
        assert_eq!(new_vector2.get_many(&[]), vec![]);

        // Test `get_all` for a restored DB
        assert_eq!(
            vec![
                S([1u8].to_vec()),
                S([3u8].to_vec()),
                S([3u8].to_vec()),
                S([7u8].to_vec()),
            ],
            new_vector2.get_all(),
            "`get_all` must return expected values, before mutation"
        );
        new_vector2.set(1, S([130u8].to_vec()));
        assert_eq!(
            vec![
                S([1u8].to_vec()),
                S([130u8].to_vec()),
                S([3u8].to_vec()),
                S([7u8].to_vec()),
            ],
            new_vector2.get_all(),
            "`get_all` must return expected values, after mutation"
        );
    }

    #[should_panic(
        expected = "Out-of-bounds. Got 2 but length was 2. persisted vector name: test-vector"
    )]
    #[test]
    fn out_of_bounds_using_get() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        let mut rusty_storage = SimpleRustyStorage::new(db);
        let mut vector = rusty_storage.schema.new_vec::<u64, u64>("test-vector");

        // initialize
        rusty_storage.restore_or_new();

        vector.push(1);
        vector.push(1);
        vector.get(2);
    }

    #[should_panic(
        expected = "Out-of-bounds. Got indices [0, 0, 0, 1, 1, 2] but length was 2. persisted vector name: test-vector"
    )]
    #[test]
    fn out_of_bounds_using_get_many() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        let mut rusty_storage = SimpleRustyStorage::new(db);
        let mut vector = rusty_storage.schema.new_vec::<u64, u64>("test-vector");

        // initialize
        rusty_storage.restore_or_new();

        vector.push(1);
        vector.push(1);
        vector.get_many(&[0, 0, 0, 1, 1, 2]);
    }

    #[should_panic(
        expected = "Out-of-bounds. Got 1 but length was 1. persisted vector name: test-vector"
    )]
    #[test]
    fn out_of_bounds_using_set_many() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        let mut rusty_storage = SimpleRustyStorage::new(db);
        let mut vector = rusty_storage.schema.new_vec::<u64, u64>("test-vector");

        // initialize
        rusty_storage.restore_or_new();

        vector.push(1);

        // attempt to set 2 values, when only one is in vector.
        vector.set_many([(0, 0), (1, 1)]);
    }

    #[should_panic(expected = "size-mismatch.  input has 2 elements and target has 1 elements")]
    #[test]
    fn size_mismatch_too_many_using_set_all() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        let mut rusty_storage = SimpleRustyStorage::new(db);
        let mut vector = rusty_storage.schema.new_vec::<u64, u64>("test-vector");

        // initialize
        rusty_storage.restore_or_new();

        vector.push(1);

        // attempt to set 2 values, when only one is in vector.
        vector.set_all(&[0, 1]);
    }

    #[should_panic(expected = "size-mismatch.  input has 1 elements and target has 2 elements")]
    #[test]
    fn size_mismatch_too_few_using_set_all() {
        let opt = rusty_leveldb::in_memory();
        let db = DB::open("test-database", opt.clone()).unwrap();

        let mut rusty_storage = SimpleRustyStorage::new(db);
        let mut vector = rusty_storage.schema.new_vec::<u64, u64>("test-vector");

        // initialize
        rusty_storage.restore_or_new();

        vector.push(0);
        vector.push(1);

        // attempt to set 1 values, when two are in vector.
        vector.set_all(&[5]);
    }

    pub mod concurrency {
        use super::super::StorageVec;
        use super::*;
        use std::thread;

        type TestVec = Arc<Mutex<DbtVec<RustyKey, RustyValue, Index, u64>>>;

        // pub fn prepare_concurrency_test_singleton(singleton: &impl StorageSingleton<u64>) {
        //     singleton.set(42);
        // }

        fn gen_concurrency_test_vecs(num: u8) -> Vec<TestVec> {
            let opt = rusty_leveldb::in_memory();
            let db = DB::open("test-database", opt.clone()).unwrap();
            let mut rusty_storage = SimpleRustyStorage::new(db);

            (0..num)
                .map(|i| {
                    let mut vec = rusty_storage
                        .schema
                        .new_vec::<Index, u64>(&format!("atomicity-test-vector #{}", i));
                    for j in 0u64..300 {
                        vec.push(j);
                    }
                    vec
                })
                .collect()
        }

        pub fn iter_all_eq<T: PartialEq>(iter: impl IntoIterator<Item = T>) -> bool {
            let mut iter = iter.into_iter();
            let first = match iter.next() {
                Some(f) => f,
                None => return true,
            };
            iter.all(|elem| elem == first)
        }

        // This test fails, which demonstrates that concurrent reads and writes
        // to multiple DbtVec in a `DbtSchema` are not atomic when the DbtVec are
        // individually shared between threads.
        //
        // This failure is expected (correct), but somewhat subtle and not
        // immediately obvious if one is expecting atomic read/write guarantee
        // across all `Tables` created by a single `DbtSchema` instance in all cases.
        //
        // The construction shared between threads is:
        //   Vec <Arc<Mutex<  Arc<Mutex< DbtVec<u64> >> >> >
        //
        // Alternatively, these could be individually shared between threads:
        //   <Arc<Mutex<  Arc<Mutex< DbtVec<u64> >> >>
        //   <Arc<Mutex<  Arc<Mutex< DbtVec<u64> >> >>
        //
        // The test fails which demonstrates that this construction does NOT
        // provide atomic read/write guarantee over multiple tables.
        //
        // Recommendations:
        //  1. Improve the twenty-first API and doc-comments to discourage
        //     this usage and facilitate/promote an atomic usage.
        //  2. Audit neptune-core and triton-vm to ensure this usage does
        //     not occur anywhere.
        #[test]
        pub fn multi_table_atomic_setmany_and_getmany_immutable() {
            let num_vecs = 2;

            use rand::{seq::SliceRandom, thread_rng};
            let opt = rusty_leveldb::in_memory();
            let db = DB::open("test-database", opt.clone()).unwrap();
            let mut rusty_storage = SimpleRustyStorage::new(db);

            let table_vecs = (0..num_vecs)
                .map(|i| {
                    let mut vec = rusty_storage
                        .schema
                        .new_vec::<Index, u64>(&format!("atomicity-test-vector #{}", i));
                    for i in 0u64..300 {
                        vec.push(i);
                    }

                    // StorageVec is impl'd on Arc<Mutex<DbtVec>>.  But StorageVec contains
                    // `&self mut` methods, so an Arc<Mutex<DbtVec>> that calls a StorageVec
                    // mutable method is not shareable between threads.  The solution is to
                    // make these vecs Immutable by wrapping in extra Arc<Mutex>>
                    // which allows them to be mutably shared between threads
                    //
                    // Note that this is what neptune-core actually does in practice!
                    //
                    // Note also that without doing this, DbtVec is unusable across threads!!
                    // Double-locking is an inefficiency and code-smell.  The correct fix is
                    // to change StorageVec mutable (set_*, push, pop) to take `&self` instead of
                    // &mut self.  The refactored code does this.
                    Arc::new(Mutex::new(vec))
                })
                .collect_vec();

            let orig = table_vecs[0].lock().unwrap().get_all();
            let modified: Vec<u64> = orig.iter().map(|_| 50).collect();

            // note: all vecs have the same length and same values.
            let indices: Vec<_> = (0..orig.len() as u64).collect();

            // We loop X times, and for each loop we start a `gets` reader thread and a `sets`
            // writer thread.  The data in all 3 vecs is identical at start of each iteration.
            // The writer thread writes data to all 3 vecs in random order, and the reader thread
            // reads from them in random order.  The reader thread checks that each vec is internally
            // consistent and also that all 3 vecs match eachother.  If either check fails,
            // we have inconsistency across `Dbtable`s and the test fails.
            //
            // Note: this test eventually fails even if tables are always written and read in the
            // same order. However it fails *much* faster with a random order, typically in a
            // few iterations.  For deterministic ordering, comment the two calls to vecs.shuffle()
            for i in 0..10000 {
                thread::scope(|s| {
                    // Reader thread.  Calls get_many() on each DbtVec "table" reading all elements
                    let gets = s.spawn(|| {
                        let mut vecs = table_vecs.iter().collect_vec();
                        vecs.shuffle(&mut thread_rng());

                        let mut copies = vec![];
                        for vec in vecs.iter() {

                            // perform a read of all values
                            let copy = (*vec).lock().unwrap().get_many(&indices);

                            // We expect this assert can never fail.  It would mean an
                            // individual DbtVec instance is internally inconsistent, but we know
                            // all reads/writes are protected by a lock.
                            assert!(
                                copy == orig || copy == modified,
                                "encountered inconsistent table read. data: {:?}",
                                copy
                            );

                            copies.push(copy);
                        }

                        let sums: Vec<u64> = copies.iter().map(|f| f.iter().sum()).collect();

                        // This assert tests that the two vecs have equal sums.  If not, then one
                        // was changed by the writer thread in between our calls to get_many().
                        //
                        // With true atomic writes over multiple tables, this would never fail,
                        // but in our setup it can because locks only exist over individual tables.
                        assert!(
                            iter_all_eq(sums.clone()),
                            "in test iteration #{} encountered inconsistent read across tables:\n  sums: {:?}\n  data: {:?}",
                            i,
                            sums,
                            copies
                        );
                    });

                    // Writer thread.  Calls set_many() on each DbtVec "table" modifying all elements
                    let sets = s.spawn(|| {
                        let mut vecs = table_vecs.iter().collect_vec();
                        vecs.shuffle(&mut thread_rng());

                        for vec in vecs.iter() {
                            vec.lock()
                                .unwrap()
                                .set_many(orig.iter().enumerate().map(|(k, _v)| (k as u64, 50u64)));
                        }
                    });
                    gets.join().unwrap();
                    sets.join().unwrap();

                    // reset all values back to original state in each DbtVec/table.
                    for vec in table_vecs.iter() {
                        vec.lock().unwrap().set_all(&orig);
                    }
                });
            }
        }

        // This test demonstrates that concurrent reads and writes
        // to multiple DbtVec in a `DbtSchema` are atomic when
        // a lock is placed around a collection holding all the `DbtVec`.
        //
        // The construction shared between threads is:
        //   <Arc<Mutex<  Vec< Arc<Mutex< DbtVec<u64> >> > >>
        //
        // The test passes and demonstrates that this construction provides
        // atomic read/write guarantee over multiple tables.
        //
        // It is recommended then, that twenty-first be enhanced to offer
        // a simple API around this construction to document and facilitate
        // its use and discourage (or even prevent?) the non-atomic use.
        #[test]
        pub fn multi_table_mutex_atomic_setmany_and_getmany() {
            use rand::{seq::SliceRandom, thread_rng};

            let table_vecs = gen_concurrency_test_vecs(2);
            let locked_table_vecs = Arc::new(Mutex::new(table_vecs));

            let orig = locked_table_vecs.lock().unwrap()[0].get_all();
            let modified: Vec<u64> = orig.iter().map(|_| 50).collect();

            // note: all vecs have the same length and same values.
            let indices: Vec<_> = (0..orig.len() as u64).collect();

            // this test should never fail.  we only loop 100 times to keep
            // the test fast.  Bump it up to 10000+ temporarily to be extra certain.
            thread::scope(|s| {
                for _i in 0..100 {
                    let gets = s.spawn(|| {
                        let mut copies = vec![];
                        for vec in locked_table_vecs.lock().unwrap().iter() {
                            let copy = vec.get_many(&indices);
                            thread::sleep(std::time::Duration::from_millis(1));

                            assert!(
                                copy == orig || copy == modified,
                                "encountered inconsistent table read. data: {:?}",
                                copy
                            );

                            copies.push(copy);
                        }

                        let sums: Vec<u64> = copies.iter().map(|f| f.iter().sum()).collect();
                        assert!(
                            iter_all_eq(sums.clone()),
                            "encountered inconsistent read across tables:\n  sums: {:?}\n  data: {:?}",
                            sums,
                            copies
                        );
                        println!("sums: {:?}", sums);
                    });

                    let sets = s.spawn(|| {
                        let mut lock = locked_table_vecs.lock().unwrap();

                        let mut vecs = lock.iter_mut().collect_vec();
                        vecs.shuffle(&mut thread_rng());

                        for i in 0..vecs.len() {
                            vecs.get_mut(i)
                                .unwrap()
                                .set_many(orig.iter().enumerate().map(|(k, _v)| (k as u64, 50u64)));
                        }
                    });
                    gets.join().unwrap();
                    sets.join().unwrap();

                    println!("--- threads finished. restart. ---");

                    for vec in locked_table_vecs.lock().unwrap().iter_mut() {
                        vec.set_all(&orig);
                    }
                }
            });
        }

        // This test fails to compile because Arc<Mutex<DbtVec<u64>>>
        // cannot be shared beteen threads.  This is because
        // Arc<Mutex<DbtVec<T>>> impl's StorageVec, which uses
        // `&mut self` for all write operation methods.
        //
        // That is unnecessary, as Arc<Mutex<..>> is immutable, and indeed
        // Rust guarantees/requires that only immutable `struct` can be
        // safely passed between threads.
        //
        // In other words, this test that won't compile demonstrates
        // how/why StorageVec trait needs to be modified to take only `&self`.
        //
        // Indeed, with that StorageVec change in place, the other two tests
        // above would not need the outer Arc<Mutex< .. >> wrapper.

        // #[test]
        // pub fn multi_table_atomic_setmany_and_getmany_cannot_compile() {
        //     let opt = rusty_leveldb::in_memory();
        //     let db = DB::open("test-database", opt.clone()).unwrap();
        //     let rusty_storage = Arc::new(Mutex::new(SimpleRustyStorage::new(db)));

        //     let mut vecs = (0..2)
        //         .map(|i| {
        //             let mut vec = rusty_storage
        //                 .lock()
        //                 .unwrap()
        //                 .schema
        //                 .new_vec::<Index, u64>(&format!("atomicity-test-vector #{}", i));
        //             for i in 0u64..300 {
        //                 vec.push(i);
        //             }
        //             vec
        //         })
        //         .collect_vec();

        //     let vec1 = vecs.pop().unwrap();
        //     let vec2 = vecs.pop().unwrap();

        //     let orig = vec1.get_all();
        //     let modified: Vec<u64> = orig.iter().map(|_| 50).collect();

        //     // note: all vecs have the same length and same values.
        //     let indices: Vec<_> = (0..orig.len() as u64).collect();

        //     // this test should never fail.  we only loop 100 times to keep
        //     // the test fast.  Bump it up to 10000+ temporarily to be extra certain.
        //     thread::scope(|s| {
        //         for _i in 0..100 {
        //             let gets = s.spawn(|| {
        //                 let vecs = vec![&vec1, &vec2];
        //                 let mut copies = vec![];
        //                 for vec in vecs.iter() {

        //                     let copy = (*vec).get_many(&indices);
        //                     thread::sleep(std::time::Duration::from_millis(1));

        //                     assert!(
        //                         copy == orig || copy == modified,
        //                         "encountered inconsistent table read. data: {:?}",
        //                         copy
        //                     );

        //                     copies.push(copy);
        //                 }

        //                 let sums: Vec<u64> = copies.iter().map(|f| f.iter().sum()).collect();
        //                 assert!(
        //                     iter_all_eq(sums.clone()),
        //                     "encountered inconsistent read across tables:\n  sums: {:?}\n  data: {:?}",
        //                     sums,
        //                     copies
        //                 );
        //                 println!("sums: {:?}", sums);
        //             });

        //             let sets = s.spawn(|| {
        //                 // prevents test from compiling
        //                 vec1.set_many(orig.iter().enumerate().map(|(k, _v)| (k as u64, 50u64)));
        //                 vec2.set_many(orig.iter().enumerate().map(|(k, _v)| (k as u64, 50u64)));
        //             });
        //             gets.join().unwrap();
        //             sets.join().unwrap();

        //             println!("--- threads finished. restart. ---");

        //             // prevents test from compiling
        //             vec1.set_all(&orig);
        //             vec2.set_all(&orig);
        //         }
        //     });
        // }
    }
}
