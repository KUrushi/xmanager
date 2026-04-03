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
"""Tests for xmanager.xm_local.packaging.databricks."""
import os
import tempfile
import unittest
from unittest import mock

from xmanager import xm
from xmanager.xm import executables
from xmanager.xm_local import executables as local_executables
from xmanager.xm_local import executors as local_executors
from xmanager.xm_local.packaging import databricks


class ExtractPackageNameTest(unittest.TestCase):

  def test_standard_wheel(self):
    self.assertEqual(
        databricks._extract_package_name(
            '/tmp/my_package-0.1.0-py3-none-any.whl'
        ),
        'my_package',
    )

  def test_complex_version(self):
    self.assertEqual(
        databricks._extract_package_name(
            '/tmp/pkg-1.2.3.post1-cp310-cp310-linux_x86_64.whl'
        ),
        'pkg',
    )


class BuildWheelTest(unittest.TestCase):

  @mock.patch('subprocess.run')
  def test_build_wheel_success(self, mock_run):
    with tempfile.TemporaryDirectory() as tmpdir:
      wheel_path = os.path.join(tmpdir, 'test_pkg-0.1.0-py3-none-any.whl')
      open(wheel_path, 'w').close()

      with mock.patch('tempfile.mkdtemp', return_value=tmpdir):
        result = databricks._build_wheel('/path/to/project')

      self.assertEqual(result, wheel_path)
      mock_run.assert_called_once_with(
          ['uv', 'build', '--wheel', '--out-dir', tmpdir, '/path/to/project'],
          check=True,
      )

  @mock.patch('subprocess.run', side_effect=FileNotFoundError)
  def test_build_wheel_uv_not_found(self, mock_run):
    with self.assertRaises(FileNotFoundError) as ctx:
      databricks._build_wheel('/path/to/project')
    self.assertIn('uv', str(ctx.exception))


class GetUploadPathTest(unittest.TestCase):

  def test_databricks_spec(self):
    spec = local_executors.DatabricksSpec(
        upload_path='dbfs:/custom/path'
    )
    self.assertEqual(databricks._get_upload_path(spec), 'dbfs:/custom/path')

  def test_unsupported_spec(self):
    with self.assertRaises(TypeError):
      databricks._get_upload_path(local_executors.LocalSpec())


class PackageDatabricksExecutableTest(unittest.TestCase):

  @mock.patch.object(databricks, '_upload_wheel')
  @mock.patch.object(databricks, '_build_wheel')
  def test_package_python_wheel(self, mock_build, mock_upload):
    mock_build.return_value = '/tmp/my_pkg-0.1.0-py3-none-any.whl'
    mock_upload.return_value = 'dbfs:/FileStore/wheels/my_pkg-0.1.0-py3-none-any.whl'

    packageable = xm.Packageable(
        executable_spec=executables.PythonWheel(
            entrypoint='train',
            path='/path/to/project',
        ),
        executor_spec=local_executors.DatabricksSpec(),
        args={'lr': 0.001},
        env_vars={'KEY': 'value'},
    )

    result = databricks.package_databricks_executable(
        {}, packageable, packageable.executable_spec
    )

    self.assertIsInstance(result, local_executables.DatabricksWheel)
    self.assertEqual(
        result.wheel_path,
        'dbfs:/FileStore/wheels/my_pkg-0.1.0-py3-none-any.whl',
    )
    self.assertEqual(result.entry_point, 'train')
    self.assertEqual(result.package_name, 'my_pkg')

  def test_unsupported_executable_spec(self):
    packageable = xm.Packageable(
        executable_spec=executables.Container(image_path='test'),
        executor_spec=local_executors.DatabricksSpec(),
    )
    with self.assertRaises(TypeError):
      databricks.package_databricks_executable(
          {}, packageable, packageable.executable_spec
      )


if __name__ == '__main__':
  unittest.main()
