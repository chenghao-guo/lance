// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Distributed vector index coordinator

use lance_core::Result;
use lance_linalg::distance::DistanceType;

use crate::vector::ivf::builder::IvfBuildParams;

use super::config::DistributedVectorIndexConfig;
use crate::vector::distributed::ivf_coordinator::DistributedIvfCoordinator;
use crate::vector::distributed::progress_tracker::{BuildPhase, ProgressTracker};
use crate::vector::distributed::quality_validator::QualityValidator;

/// Main coordinator for distributed vector index building
#[allow(dead_code)]
pub struct DistributedVectorIndexCoordinator {
    config: DistributedVectorIndexConfig,
    distance_type: DistanceType,
    dimension: usize,
    ivf_coordinator: DistributedIvfCoordinator,
    quality_validator: QualityValidator,
    progress_tracker: ProgressTracker,
}

impl DistributedVectorIndexCoordinator {
    /// Create a new distributed vector index coordinator
    pub fn new(
        config: DistributedVectorIndexConfig,
        distance_type: DistanceType,
        dimension: usize,
    ) -> Self {
        let ivf_config = config.ivf_config.base_params.clone();

        let ivf_coordinator = DistributedIvfCoordinator::new(ivf_config, 1);
        let quality_validator = QualityValidator::new();
        let progress_tracker = ProgressTracker::new(1, 1000000);

        Self {
            config,
            distance_type,
            dimension,
            ivf_coordinator,
            quality_validator,
            progress_tracker,
        }
    }

    /// Build distributed IVF index
    pub async fn build_distributed_ivf(
        &self,
        fragments: Vec<super::ivf_coordinator::Fragment>,
        column: String,
        num_partitions: usize,
    ) -> Result<()> {
        self.progress_tracker.update_phase(BuildPhase::IvfTraining);

        log::info!("Starting distributed IVF index building");
        log::info!("Configuration: {:?}", self.config);
        log::info!(
            "Distance type: {:?}, Dimension: {}",
            self.distance_type,
            self.dimension
        );
        log::info!(
            "Building IVF index with {} fragments on column {}",
            fragments.len(),
            column
        );

        // Phase 1: Distributed IVF training
        let mut ivf_coordinator = DistributedIvfCoordinator::new(
            self.get_adjusted_ivf_params(num_partitions),
            fragments.len(),
        );

        // Calculate total dataset size
        let total_dataset_size: usize = fragments.iter().map(|frag| frag.row_count).sum();
        ivf_coordinator.set_total_dataset_size(total_dataset_size);

        // Run distributed IVF training
        let ivf_model = ivf_coordinator
            .train_distributed_ivf(
                &fragments,
                &column, // column name
                num_partitions,
                256, // sample rate - this should be calculated based on actual data
            )
            .await?;

        log::info!(
            "IVF training completed with {} partitions",
            ivf_model.num_partitions()
        );

        self.progress_tracker.mark_completed();

        log::info!("Distributed IVF index building completed successfully!");

        Ok(())
    }

    /// Get adjusted IVF parameters for distributed training
    pub fn get_adjusted_ivf_params(&self, num_partitions: usize) -> IvfBuildParams {
        let mut params = self.config.ivf_config.base_params.clone();

        // Apply distributed adjustments
        params.sample_rate =
            (params.sample_rate as f64 * self.config.ivf_config.sample_rate_multiplier) as usize;
        params.max_iters += self.config.ivf_config.max_iters_bonus;
        params.num_partitions = Some(num_partitions);

        params
    }
}

/// Fragment data representation for distributed processing
#[derive(Debug, Clone)]
pub struct FragmentData {
    pub fragment_id: usize,
    pub row_count: usize,
    pub data_path: String,
}

impl FragmentData {
    /// Create new fragment data
    pub fn new(fragment_id: usize, row_count: usize, data_path: String) -> Self {
        Self {
            fragment_id,
            row_count,
            data_path,
        }
    }
}
