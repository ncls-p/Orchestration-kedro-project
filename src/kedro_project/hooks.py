from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession


class SparkHooks:
    @hook_impl
    def after_context_created(self, context: KedroContext) -> None:
        """Initialises a SparkSession using the config
        defined in project's conf folder.
        """

        # Load the spark configuration in spark.yaml using the config loader
        parameters = context.config_loader["spark"]
        # Convert parameters to list of (key, value) pairs and ignore type mismatch
        spark_conf = SparkConf().setAll(list(parameters.items()))

        # Initialise the spark session
        # Get project name as a string
        project_name = str(context.project_path.name)

        # Build spark session step by step to avoid type issues
        spark_session_conf = (
            getattr(SparkSession.builder, "appName")(project_name)
            .enableHiveSupport()
            .config(conf=spark_conf)
        )
        _spark_session = spark_session_conf.getOrCreate()
        _spark_session.sparkContext.setLogLevel("WARN")
