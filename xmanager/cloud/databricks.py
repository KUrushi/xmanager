# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Client for interacting with Databricks.

Uses the Databricks SDK for authentication (OAuth M2M) and job submission
via the runs/submit API.
"""

import asyncio
from typing import Any, Sequence

import attr
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs as databricks_jobs

from xmanager import xm
from xmanager.xm_local import executables as local_executables
from xmanager.xm_local import executors as local_executors
from xmanager.xm_local import handles
from xmanager.xm_local import registry
from xmanager.xm_local import status as local_status

_POLL_INTERVAL_SECONDS = 10

_default_client = None


def _databricks_job_predicate(job: xm.Job) -> bool:
  return isinstance(job.executor, local_executors.Databricks)


class Client:
  """Client class for interacting with Databricks."""

  def __init__(self) -> None:
    # WorkspaceClient auto-discovers auth from environment variables:
    # OAuth M2M: DATABRICKS_HOST, DATABRICKS_CLIENT_ID,
    #            DATABRICKS_CLIENT_SECRET
    # PAT:       DATABRICKS_HOST, DATABRICKS_TOKEN
    self.workspace = WorkspaceClient()

  def submit_run(
      self,
      run_name: str,
      jobs: Sequence[xm.Job],
  ) -> int:
    """Submit a one-shot run using the runs/submit API."""
    tasks = []
    for i, job in enumerate(jobs):
      executable = job.executable
      if not isinstance(executable, local_executables.DatabricksWheel):
        raise ValueError(
            f'Executable {executable} has type {type(executable)}. '
            'Executable must be of type DatabricksWheel.'
        )

      executor = job.executor
      assert isinstance(executor, local_executors.Databricks)

      merged_args = xm.merge_args(executable.args, job.args)
      named_parameters = {
          k: str(v)
          for k, v in merged_args.to_dict(kwargs_only=True).items()
      }

      all_env_vars = {**executable.env_vars, **job.env_vars}

      new_cluster = {
          'spark_version': executor.spark_version,
          'node_type_id': executor.node_type_id,
          'num_workers': executor.num_workers,
      }
      if all_env_vars:
        new_cluster['spark_env_vars'] = {
            k: str(v) for k, v in all_env_vars.items()
        }
      if executor.cluster_config:
        new_cluster.update(executor.cluster_config)

      task_key = job.name or f'task_{i}'

      task = databricks_jobs.SubmitTask(
          task_key=task_key,
          new_cluster=databricks_jobs.ClusterSpec(**new_cluster),
          python_wheel_task=databricks_jobs.PythonWheelTask(
              package_name=executable.package_name,
              entry_point=executable.entry_point,
              named_parameters=named_parameters,
          ),
          libraries=[
              databricks_jobs.TaskLibrary(whl=executable.wheel_path),
          ],
      )
      tasks.append(task)

    response = self.workspace.jobs.submit(
        run_name=run_name,
        tasks=tasks,
    )
    run_id = response.run_id

    host = self.workspace.config.host.rstrip('/')
    print(f'Databricks run submitted: {host}/#job/0/run/{run_id}')

    return run_id

  async def wait_for_run(self, run_id: int) -> None:
    """Poll until the run reaches a terminal state."""
    while True:
      await asyncio.sleep(_POLL_INTERVAL_SECONDS)
      run = self.workspace.jobs.get_run(run_id)
      state = run.state
      life_cycle = (
          state.life_cycle_state.value if state.life_cycle_state else None
      )

      if life_cycle == 'TERMINATED':
        result = state.result_state.value if state.result_state else None
        if result == 'SUCCESS':
          return
        raise RuntimeError(
            f'Databricks run {run_id} terminated with result: {result}. '
            f'Message: {state.state_message}'
        )
      elif life_cycle in ('INTERNAL_ERROR', 'SKIPPED'):
        raise RuntimeError(
            f'Databricks run {run_id} ended with state: {life_cycle}. '
            f'Message: {state.state_message}'
        )

  def get_run_status(self, run_id: int) -> local_status.LocalWorkUnitStatus:
    """Get the current status of a run."""
    run = self.workspace.jobs.get_run(run_id)
    state = run.state
    life_cycle = (
        state.life_cycle_state.value if state.life_cycle_state else None
    )

    if life_cycle == 'TERMINATED':
      result = state.result_state.value if state.result_state else 'FAILED'
      status_enum = _RESULT_STATE_TO_STATUS.get(
          result, local_status.LocalWorkUnitStatusEnum.FAILED
      )
    else:
      status_enum = _LIFECYCLE_STATE_TO_STATUS.get(
          life_cycle, local_status.LocalWorkUnitStatusEnum.RUNNING
      )

    message = state.state_message or ''
    return local_status.LocalWorkUnitStatus(
        status=status_enum, message=message
    )


_LIFECYCLE_STATE_TO_STATUS = {
    'PENDING': local_status.LocalWorkUnitStatusEnum.RUNNING,
    'RUNNING': local_status.LocalWorkUnitStatusEnum.RUNNING,
    'TERMINATING': local_status.LocalWorkUnitStatusEnum.RUNNING,
    'BLOCKED': local_status.LocalWorkUnitStatusEnum.RUNNING,
    'WAITING_FOR_RETRY': local_status.LocalWorkUnitStatusEnum.RUNNING,
    'SKIPPED': local_status.LocalWorkUnitStatusEnum.CANCELLED,
    'INTERNAL_ERROR': local_status.LocalWorkUnitStatusEnum.FAILED,
}

_RESULT_STATE_TO_STATUS = {
    'SUCCESS': local_status.LocalWorkUnitStatusEnum.COMPLETED,
    'FAILED': local_status.LocalWorkUnitStatusEnum.FAILED,
    'TIMEDOUT': local_status.LocalWorkUnitStatusEnum.FAILED,
    'CANCELED': local_status.LocalWorkUnitStatusEnum.CANCELLED,
    'CANCELLED': local_status.LocalWorkUnitStatusEnum.CANCELLED,
    'MAXIMUM_CONCURRENT_RUNS_REACHED': local_status.LocalWorkUnitStatusEnum.FAILED,
    'EXCLUDED': local_status.LocalWorkUnitStatusEnum.CANCELLED,
    'SUCCESS_WITH_FAILURES': local_status.LocalWorkUnitStatusEnum.FAILED,
    'UPSTREAM_FAILED': local_status.LocalWorkUnitStatusEnum.FAILED,
    'UPSTREAM_CANCELED': local_status.LocalWorkUnitStatusEnum.CANCELLED,
}


def set_default_client(client: Client) -> None:
  global _default_client
  _default_client = client


def get_default_client() -> Client:
  global _default_client
  if _default_client is None:
    _default_client = Client()
  return _default_client


@attr.s(auto_attribs=True)
class DatabricksHandle(handles.ExecutionHandle):
  """A handle for referring to a launched Databricks run."""

  run_id: int

  async def wait(self) -> None:
    await get_default_client().wait_for_run(self.run_id)

  def stop(self) -> None:
    pass

  def get_status(self) -> local_status.LocalWorkUnitStatus:
    return get_default_client().get_run_status(self.run_id)

  def save_to_storage(self, experiment_id: int, work_unit_id: int) -> None:
    pass


def launch(
    experiment_title: str, work_unit_name: str, job_group: xm.JobGroup
) -> list[DatabricksHandle]:
  """Launch Databricks jobs in the job_group and return a handle."""
  jobs = xm.job_operators.collect_jobs_by_filter(
      job_group, _databricks_job_predicate
  )
  if not jobs:
    return []

  run_id = get_default_client().submit_run(
      run_name=f'{experiment_title}_{work_unit_name}',
      jobs=jobs,
  )
  return [DatabricksHandle(run_id=run_id)]


async def _async_launch(
    local_experiment_unit: Any, job_group: xm.JobGroup
) -> list[DatabricksHandle]:
  return launch(
      local_experiment_unit._experiment_title,  # pylint: disable=protected-access
      local_experiment_unit.experiment_unit_name,
      job_group,
  )


def register():
  """Registers Databricks execution logic."""
  registry.register(
      local_executors.Databricks,
      launch=_async_launch,
  )
