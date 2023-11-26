use super::super::level_db::DB;
use super::{RustyKey, RustyValue, StorageReader};
use leveldb::options::ReadOptions;

// Note: RustyReader and SimpleRustyReader appear to be exactly
// the same.  Can we remove one of them?

pub struct SimpleRustyReader {
    pub(super) db: DB,
}

impl StorageReader<RustyKey, RustyValue> for SimpleRustyReader {
    #[inline]
    fn get(&self, key: RustyKey) -> Option<RustyValue> {
        self.db
            .get(&ReadOptions::new(), &key.0)
            .unwrap()
            .map(RustyValue)
    }

    #[inline]
    fn get_many(&self, keys: &[RustyKey]) -> Vec<Option<RustyValue>> {
        keys.iter()
            .map(|key| {
                self.db
                    .get(&ReadOptions::new(), &key.0)
                    .unwrap()
                    .map(RustyValue)
            })
            .collect()
    }
}
