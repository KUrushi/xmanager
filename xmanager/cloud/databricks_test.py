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
"""Tests for xmanager.cloud.databricks."""
import unittest
from unittest import mock

from xmanager import xm
from xmanager.xm_local import executables as local_executables
from xmanager.xm_local import executors as local_executors
from xmanager.xm_local import status as local_status

from xmanager.cloud import databricks


class DatabricksClientTest(unittest.TestCase):

  @mock.patch('xmanager.cloud.databricks.WorkspaceClient')
  def test_submit_run(self, mock_workspace_cls):
    mock_workspace = mock_workspace_cls.return_value
    mock_workspace.config.host = 'https://test.cloud.databricks.com'
    mock_workspace.jobs.submit.return_value = mock.Mock(run_id=12345)

    client = databricks.Client()
    job = xm.Job(
        name='test-job',
        executable=local_executables.DatabricksWheel(
            name='test-wheel',
            wheel_path='dbfs:/FileStore/wheels/pkg-0.1.0-py3-none-any.whl',
            entry_point='train',
            package_name='pkg',
            args=xm.SequentialArgs.from_collection({'a': '1'}),
        ),
        executor=local_executors.Databricks(
            spark_version='15.4.x-scala2.12',
            node_type_id='i3.xlarge',
            num_workers=0,
        ),
        args={'b': '2'},
    )

    run_id = client.submit_run('test-experiment', [job])
    self.assertEqual(run_id, 12345)
    mock_workspace.jobs.submit.assert_called_once()
    call_kwargs = mock_workspace.jobs.submit.call_args
    self.assertEqual(call_kwargs.kwargs['run_name'], 'test-experiment')
    self.assertEqual(len(call_kwargs.kwargs['tasks']), 1)

  @mock.patch('xmanager.cloud.databricks.WorkspaceClient')
  def test_get_run_status_running(self, mock_workspace_cls):
    mock_workspace = mock_workspace_cls.return_value
    mock_state = mock.Mock()
    mock_state.life_cycle_state.value = 'RUNNING'
    mock_state.state_message = 'Running'
    mock_workspace.jobs.get_run.return_value = mock.Mock(state=mock_state)

    client = databricks.Client()
    status = client.get_run_status(12345)
    self.assertEqual(
        status._status, local_status.LocalWorkUnitStatusEnum.RUNNING
    )

  @mock.patch('xmanager.cloud.databricks.WorkspaceClient')
  def test_get_run_status_completed(self, mock_workspace_cls):
    mock_workspace = mock_workspace_cls.return_value
    mock_state = mock.Mock()
    mock_state.life_cycle_state.value = 'TERMINATED'
    mock_state.result_state.value = 'SUCCESS'
    mock_state.state_message = ''
    mock_workspace.jobs.get_run.return_value = mock.Mock(state=mock_state)

    client = databricks.Client()
    status = client.get_run_status(12345)
    self.assertEqual(
        status._status, local_status.LocalWorkUnitStatusEnum.COMPLETED
    )

  @mock.patch('xmanager.cloud.databricks.WorkspaceClient')
  def test_get_run_status_failed(self, mock_workspace_cls):
    mock_workspace = mock_workspace_cls.return_value
    mock_state = mock.Mock()
    mock_state.life_cycle_state.value = 'TERMINATED'
    mock_state.result_state.value = 'FAILED'
    mock_state.state_message = 'Task failed'
    mock_workspace.jobs.get_run.return_value = mock.Mock(state=mock_state)

    client = databricks.Client()
    status = client.get_run_status(12345)
    self.assertEqual(
        status._status, local_status.LocalWorkUnitStatusEnum.FAILED
    )
    self.assertEqual(status.message, 'Task failed')


class DatabricksLaunchTest(unittest.TestCase):

  @mock.patch.object(databricks, 'get_default_client')
  def test_launch_with_databricks_jobs(self, mock_get_client):
    mock_client = mock.Mock()
    mock_client.submit_run.return_value = 42
    mock_get_client.return_value = mock_client

    executable = local_executables.DatabricksWheel(
        name='test',
        wheel_path='dbfs:/wheels/test.whl',
        entry_point='train',
        package_name='test_pkg',
    )
    job_group = xm.JobGroup(
        test_job=xm.Job(
            executable=executable,
            executor=local_executors.Databricks(),
            args={},
        )
    )

    handles = databricks.launch('exp', 'wu_0', job_group)
    self.assertEqual(len(handles), 1)
    self.assertEqual(handles[0].run_id, 42)

  @mock.patch.object(databricks, 'get_default_client')
  def test_launch_with_no_databricks_jobs(self, mock_get_client):
    job_group = xm.JobGroup()
    handles = databricks.launch('exp', 'wu_0', job_group)
    self.assertEqual(len(handles), 0)
    mock_get_client.assert_not_called()


if __name__ == '__main__':
  unittest.main()
