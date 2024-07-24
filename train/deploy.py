from datetime import timedelta

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import IntervalSchedule
from train_flow import train_flow

interval_schedule = IntervalSchedule(interval=timedelta(minutes=5))

docker_dep = Deployment.build_from_flow(
    flow=train_flow,
    name="model-training",
    infra_overrides={"env": {"PREFECT_LOGGING_LEVEL": "DEBUG"}},
    work_queue_name="default",
    schedule=interval_schedule,
)

if __name__ == "__main__":
    docker_dep.apply()
