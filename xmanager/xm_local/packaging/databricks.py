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
"""Packaging for execution on Databricks."""

import glob
import os
import subprocess
import tempfile

from xmanager import xm
from xmanager.xm import executables
from xmanager.xm_local import executables as local_executables
from xmanager.xm_local import executors as local_executors
from xmanager.xm_local.packaging import bazel_tools


def _get_upload_path(executor_spec: xm.ExecutorSpec) -> str:
  match executor_spec:
    case local_executors.DatabricksSpec() as spec:
      return spec.upload_path
    case _:
      raise TypeError(
          f'Unsupported executor specification: {executor_spec!r}'
      )


def _build_wheel(project_path: str) -> str:
  """Build a wheel using 'uv build' and return the path to the .whl file."""
  dist_dir = tempfile.mkdtemp(prefix='xm_databricks_')
  try:
    subprocess.run(
        ['uv', 'build', '--wheel', '--out-dir', dist_dir, project_path],
        check=True,
    )
  except FileNotFoundError:
    raise FileNotFoundError(
        "'uv' command not found. Please install uv: "
        'https://docs.astral.sh/uv/getting-started/installation/'
    ) from None
  wheels = glob.glob(os.path.join(dist_dir, '*.whl'))
  if not wheels:
    raise FileNotFoundError(
        f'No wheel found after building {project_path}'
    )
  return wheels[0]


def _extract_package_name(wheel_path: str) -> str:
  """Extract the distribution name from a wheel filename."""
  # Wheel filenames follow: {name}-{ver}(-{build})?-{python}-{abi}-{plat}.whl
  basename = os.path.basename(wheel_path)
  return basename.split('-')[0]


def _upload_wheel(wheel_local_path: str, upload_prefix: str) -> str:
  """Upload a wheel to DBFS and return the DBFS path."""
  from databricks.sdk import WorkspaceClient

  w = WorkspaceClient()
  wheel_name = os.path.basename(wheel_local_path)
  dbfs_path = f'{upload_prefix}/{wheel_name}'

  with open(wheel_local_path, 'rb') as f:
    w.dbfs.put(dbfs_path.replace('dbfs:', ''), f, overwrite=True)

  return dbfs_path


def package_databricks_executable(
    bazel_outputs: bazel_tools.TargetOutputs,
    packageable: xm.Packageable,
    executable_spec: xm.ExecutableSpec,
) -> xm.Executable:
  """Package an executable for Databricks execution."""
  del bazel_outputs

  match executable_spec:
    case executables.PythonWheel() as python_wheel:
      upload_path = _get_upload_path(packageable.executor_spec)
      wheel_local_path = _build_wheel(python_wheel.path)
      dbfs_path = _upload_wheel(wheel_local_path, upload_path)
      package_name = _extract_package_name(wheel_local_path)

      return local_executables.DatabricksWheel(
          name=packageable.executable_spec.name,
          wheel_path=dbfs_path,
          entry_point=python_wheel.entrypoint,
          package_name=package_name,
          args=packageable.args,
          env_vars=packageable.env_vars,
      )
    case _:
      raise TypeError(
          'Unsupported executable specification '
          f'for Databricks packaging: {executable_spec!r}'
      )
