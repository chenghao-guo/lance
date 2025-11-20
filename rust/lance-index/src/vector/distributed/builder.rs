// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Distributed vector index builder implementation

use lance_core::Result;
use lance_linalg::distance::DistanceType;

use super::config::DistributedVectorIndexConfig;
use super::coordinator::{DistributedVectorIndexCoordinator, FragmentData};

/// Main builder interface for distributed vector index building
pub struct DistributedVectorIndexBuilder {
    config: DistributedVectorIndexConfig,
    distance_type: DistanceType,
    dimension: usize,
}

impl DistributedVectorIndexBuilder {
    /// Create a new distributed vector index builder
    pub fn new(
        config: DistributedVectorIndexConfig,
        distance_type: DistanceType,
        dimension: usize,
    ) -> Self {
        Self {
            config,
            distance_type,
            dimension,
        }
    }

    /// Build distributed IVF index
    pub async fn build_distributed_ivf(
        &self,
        fragments: &[FragmentData],
        column: &str,
        num_partitions: usize,
    ) -> Result<()> {
        let coordinator = DistributedVectorIndexCoordinator::new(
            self.config.clone(),
            self.distance_type,
            self.dimension,
        );

        log::info!(
            "Building distributed IVF index with {} fragments on column {} with {} partitions",
            fragments.len(),
            column,
            num_partitions
        );

        // Convert FragmentData to ivf_coordinator::Fragment
        let ivf_fragments: Vec<super::ivf_coordinator::Fragment> = fragments
            .iter()
            .map(|frag| super::ivf_coordinator::Fragment {
                id: frag.fragment_id,
                data_path: frag.data_path.clone(),
                row_count: frag.row_count,
                sample_override: None,
            })
            .collect();

        // Start the building process
        coordinator
            .build_distributed_ivf(ivf_fragments, column.to_string(), num_partitions)
            .await
    }

    /// Get the configuration
    pub fn config(&self) -> &DistributedVectorIndexConfig {
        &self.config
    }

    /// Get the distance type
    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

// Key types are available through module structure
