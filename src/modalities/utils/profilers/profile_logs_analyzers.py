import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from modalities.utils.profilers.modalities_profiler import ProfilingResult


class ProfileLogsAnalyzer:
    @staticmethod
    def load_profiling_logs(log_dir_path: Path) -> list[ProfilingResult]:
        """
        Loads the profiling logs from the specified directory.

        Args:
            log_dir_path (Path): The path to the directory containing the profiling logs.

        Returns:
            list[Result]: A list of profiling results.
        """
        results = []
        for file in log_dir_path.glob("*.json"):
            with open(file, "r") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    raise Exception(f"Could not load json file {file}") from e
            result = ProfilingResult(
                grid_search_config=data["grid_search_config"],
                env_info=data["env_info"],
                measurement=ProfilingResult.Measurement(**data["measurement"]),
                error=data.get("error", ""),
            )
            results.append(result)
        return results

    @staticmethod
    def to_pandas_df(results: list[ProfilingResult]) -> pd.DataFrame:
        """
        Converts the profiling results to a pandas DataFrame.

        Args:
            results (list[Result]): The list of profiling results.

        Returns:
            pd.DataFrame: A DataFrame containing the profiling results.
        """
        data = []
        for result in results:
            result_dict = asdict(result)
            # Flatten the 'measurement' dict into the top-level dict
            measurement = result_dict.pop("measurement", {})
            # Flatten the 'grid_search_config' dict into the top-level dict
            grid_search_config = result_dict.pop("grid_search_config", {})
            # Flatten the 'env_info' dict into the top-level dict
            env_info = result_dict.pop("env_info", {})
            flat_result = {**grid_search_config, **env_info, **measurement, **result_dict}
            data.append(flat_result)
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    log_dir = Path(
        "/raid/s3/opengptx/max_lue/repositories/modalities/tests/training/benchmark/2025-04-24__18-18-58_ed5e5044"
    )
    results = ProfileLogsAnalyzer.load_profiling_logs(log_dir)
    df = ProfileLogsAnalyzer.to_pandas_df(results)
    df["total_step_time"] = df["forward_time"] + df["backward_time"] + df["step_time"]
    df.sort_values(by=["total_step_time"], inplace=True, ascending=True)
    df["error"] = df["error"].apply(lambda x: x[:20])
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None, "display.max_colwidth", None
    ):
        print(df)
