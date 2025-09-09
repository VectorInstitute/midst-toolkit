from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from midst_toolkit.evaluation.quality.synthcity.dataloader import DataLoader
from midst_toolkit.evaluation.quality.synthcity.metric import MetricEvaluator


class StatisticalEvaluator(MetricEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        """Base class for statistical evaluators to inherit from."""
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        """Type of evaluator."""
        return "stats"

    @abstractmethod
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> dict: ...

    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> dict:
        """
        Performs evaluation using the ground truth and synthetic datasets as dataloaders by calling the internal
        ``_evaluate`` function of inheriting classes.

        Args:
            X_gt: Dataloader with ground truth (real) data.
            X_syn: Dataloader with synthetically generated data.

        Returns:
            A dictionary of results from the evaluation
        """
        return self._evaluate(X_gt, X_syn)

    def evaluate_default(
        self,
        x_gt: DataLoader,
        x_syn: DataLoader,
    ) -> float:
        """Perform a default evaluation if one is not specified."""
        return self.evaluate(x_gt, x_syn)[self._default_metric]


class AlphaPrecision(StatisticalEvaluator):
    def __init__(self, **kwargs: Any) -> None:
        """
        Evaluates the alpha-precision, beta-recall, and authenticity scores.

        The class evaluates the synthetic data using a tuple of three metrics:

            alpha-precision, beta-recall, and authenticity.

        Note that these metrics can be evaluated for each synthetic data point (which are useful for auditing and
        post-processing). Here we average the scores to reflect the overall quality of the data.
        The formal definitions can be found in the reference below:

        Alaa, Ahmed, Boris Van Breugel, Evgeny S. Saveliev, and Mihaela van der Schaar. "How faithful is your synthetic
        data? sample-level metrics for evaluating and auditing generative models."
        In International Conference on Machine Learning, pp. 290-306. PMLR, 2022.
        """
        super().__init__(default_metric="authenticity_OC", **kwargs)

    @staticmethod
    def name() -> str:
        """Return name."""
        return "alpha_precision"

    @staticmethod
    def direction() -> str:
        """Return optimization direction."""
        return "maximize"

    def metrics(
        self,
        x: np.ndarray,
        x_syn: np.ndarray,
        emb_center: np.ndarray | None = None,
    ) -> tuple[list[float], list[float], list[float], float, float, float]:
        """
        Compute the alpha-precision, beta-recall, and authenticity scores provided real data (x) and synthetic
        data (x_syn). If ``emb_center`` is provided this are "non-naive" metrics. If it is none, these constitute
        "naive" scores.

        Args:
            x: Real data
            x_syn: Synthetically generated data.
            emb_center: Center for the embeddings of the data. If None, we just use the mean of the features of x).
                Defaults to None.

        Raises:
            RuntimeError: Raised if the datasets are not the same sizes.
            RuntimeError: Raised if there is an invalid score for delta_precision_alpha.
            RuntimeError: Raised if there is an invalid score for delta_coverage_beta.

        Returns:
            alphas, alpha_precision_curve, beta_coverage_curve, delta_precision_alpha, delta_coverage_beta,
            authenticity.
        """
        if len(x) != len(x_syn):
            raise RuntimeError("The real and synthetic data must have the same length")

        if emb_center is None:
            emb_center = np.mean(x, axis=0)

        n_steps = 30
        alphas = np.linspace(0, 1, n_steps)

        radii = np.quantile(np.sqrt(np.sum((x - emb_center) ** 2, axis=1)), alphas)

        synth_center = np.mean(x_syn, axis=0)

        alpha_precision_curve: list[float] = []
        beta_coverage_curve: list[float] = []

        synth_to_center = np.sqrt(np.sum((x_syn - emb_center) ** 2, axis=1))

        nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(x)
        k_neighbors_real = nbrs_real.kneighbors(x)
        assert isinstance(k_neighbors_real, tuple)
        real_to_real, _ = k_neighbors_real

        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(x_syn)
        k_neighbors_synth = nbrs_synth.kneighbors(x)
        assert isinstance(k_neighbors_synth, tuple)
        real_to_synth, real_to_synth_args = k_neighbors_synth

        # Let us find closest real point to any real point, excluding itself (therefore 1 instead of 0)
        real_to_real = real_to_real[:, 1].squeeze()
        real_to_synth = real_to_synth.squeeze()
        real_to_synth_args = real_to_synth_args.squeeze()

        real_synth_closest = x_syn[real_to_synth_args]

        real_synth_closest_d = np.sqrt(np.sum((real_synth_closest - synth_center) ** 2, axis=1))
        closest_synth_radii = np.quantile(real_synth_closest_d, alphas)

        for k in range(len(radii)):
            precision_audit_mask = synth_to_center <= radii[k]
            alpha_precision = np.mean(precision_audit_mask)

            beta_coverage = np.mean(
                ((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_radii[k]))
            )

            alpha_precision_curve.append(alpha_precision)
            beta_coverage_curve.append(beta_coverage)

        # See which one is bigger

        authen = real_to_real[real_to_synth_args] < real_to_synth
        authenticity = np.mean(authen)

        delta_precision_alpha = 1.0 - np.sum(np.abs(np.array(alphas) - np.array(alpha_precision_curve))) / np.sum(
            alphas
        )

        if delta_precision_alpha < 0:
            raise RuntimeError("negative value detected for Delta_precision_alpha")

        delta_coverage_beta = 1.0 - np.sum(np.abs(np.array(alphas) - np.array(beta_coverage_curve))) / np.sum(alphas)

        if delta_coverage_beta < 0:
            raise RuntimeError("negative value detected for Delta_coverage_beta")

        return (
            alphas.tolist(),
            alpha_precision_curve,
            beta_coverage_curve,
            delta_precision_alpha,
            delta_coverage_beta,
            authenticity.astype(float),
        )

    def _normalize_covariates(
        self,
        x: DataLoader,
        x_syn: DataLoader,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        This is an internal method to replicate the old, naive method for evaluating AlphaPrecision.

        Args:
            x (DataLoader): The ground truth dataset.
            x_syn (DataLoader): The synthetic dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: normalized version of the datasets
        """
        x_gt_norm = x.dataframe().copy()
        x_syn_norm = x_syn.dataframe().copy()
        if self._task_type != "survival_analysis":
            if hasattr(x, "target_column"):
                x_gt_norm = x_gt_norm.drop(columns=[x.target_column])
            if hasattr(x_syn, "target_column"):
                x_syn_norm = x_syn_norm.drop(columns=[x_syn.target_column])
        scaler = MinMaxScaler().fit(x_gt_norm)
        if hasattr(x, "target_column"):
            x_gt_norm_df = pd.DataFrame(
                scaler.transform(x_gt_norm),
                columns=[col for col in x.train().dataframe().columns if col != x.target_column],
            )
        else:
            x_gt_norm_df = pd.DataFrame(scaler.transform(x_gt_norm), columns=x.train().dataframe().columns)

        if hasattr(x_syn, "target_column"):
            x_syn_norm_df = pd.DataFrame(
                scaler.transform(x_syn_norm),
                columns=[col for col in x_syn.dataframe().columns if col != x_syn.target_column],
            )
        else:
            x_syn_norm_df = pd.DataFrame(scaler.transform(x_syn_norm), columns=x_syn.dataframe().columns)

        return x_gt_norm_df, x_syn_norm_df

    def _evaluate(
        self,
        x: DataLoader,
        x_syn: DataLoader,
    ) -> dict:
        """
        Run the full evaluation pipeline, including both naive and non-naive metrics.

        Args:
            x (DataLoader): The ground truth dataset.
            x_syn (DataLoader): The synthetic dataset.

        Returns:
            Dictionary of metric type and value
        """
        results = {}

        x_ = x.numpy().reshape(len(x), -1)
        x_syn_ = x_syn.numpy().reshape(len(x_syn), -1)

        # OneClass representation
        emb = "_OC"
        oneclass_model = self._get_oneclass_model(x_)
        x_ = self._oneclass_predict(oneclass_model, x_)
        x_syn_ = self._oneclass_predict(oneclass_model, x_syn_)
        emb_center = oneclass_model.c.detach().cpu().numpy()

        (
            _,
            _,
            _,
            delta_precision_alpha,
            delta_coverage_beta,
            authenticity,
        ) = self.metrics(x_, x_syn_, emb_center=emb_center)

        results[f"delta_precision_alpha{emb}"] = delta_precision_alpha
        results[f"delta_coverage_beta{emb}"] = delta_coverage_beta
        results[f"authenticity{emb}"] = authenticity

        X_df, X_syn_df = self._normalize_covariates(x, x_syn)
        (
            _,
            _,
            _,
            delta_precision_alpha_naive,
            delta_coverage_beta_naive,
            authenticity_naive,
        ) = self.metrics(X_df.to_numpy(), X_syn_df.to_numpy(), emb_center=None)

        results["delta_precision_alpha_naive"] = delta_precision_alpha_naive
        results["delta_coverage_beta_naive"] = delta_coverage_beta_naive
        results["authenticity_naive"] = authenticity_naive

        return results
