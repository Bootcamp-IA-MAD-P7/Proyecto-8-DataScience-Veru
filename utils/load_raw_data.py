import io
import urllib.request
import polars as pl
import pyarrow.csv as pacsv
import pandas as pd
from datasets import load_dataset


class RawData:
    def __init__(
        self,
        year: int,
        base_url: str = "https://huggingface.co/datasets/Anahia/stackoverflow_survey/",
    ):
        self.year = year
        self.base_url = base_url

    def download(self) -> pl.DataFrame:
        url = f"{self.base_url}resolve/main/stackoverflow_survey_{self.year}.csv"

        try:
            print(f"Downloading dataframe {self.year}")
            df = pl.read_csv(url)
            print(f"Downloaded {df.height} rows and {df.width} columns")

            return df

        except Exception as e:
            print(f"Polars read_csv encountered issue ({e}), attempting PyArrow/Pandas fallback...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp:
                    content = resp.read()

                parse_options = pacsv.ParseOptions(
                    newlines_in_values=True, invalid_row_handler=lambda r: "skip"
                )
                with io.BytesIO(content) as buf:
                    table = pacsv.read_csv(buf, parse_options=parse_options)
                df = pl.from_arrow(table)
                print(f"Downloaded {df.height} rows and {df.width} columns via fallback")

                return df
            except Exception as fallback_e:
                try:
                    with io.BytesIO(content) as buf:
                        pdf = pd.read_csv(buf, engine="python", on_bad_lines="skip")
                    df = pl.from_pandas(pdf)
                    print(f"Downloaded {df.height} rows and {df.width} columns via pandas fallback")

                    return df
                except Exception:
                    print(f"Failed to download DataFrame {self.year}")
                    print(f"Reason: {fallback_e}")
                    raise


class Schema:
    def __init__(
        self,
        year: int,
        base_url: str = "https://huggingface.co/datasets/Anahia/stackoverflow_survey_schemas/",
    ):
        self.year = year
        self.base_url = base_url

    def download(self) -> pl.DataFrame:
        url = f"{self.base_url}resolve/main/schema_{self.year}.csv"

        try:
            print(f"Downloading schema {self.year}")
            schema = pl.read_csv(url)
            print("Schema loaded")

            return schema

        except Exception as e:
            print(f"Failed to download schema {self.year}")
            print(f"Reason: {e}")
            raise

    def load(self):
        try:
            print(f"Loading schema {self.year}")
            schema = load_dataset(
                "Anahia/stackoverflow_survey_schemas",
                data_files=f"schema_{self.year}.csv",
                streaming=True,
            ).with_format("polars")
            print("Schema loaded")

            return schema

        except Exception as e:
            print(f"Failed to stream schema {self.year}")
            print(f"Reason: {e}")
            raise
