# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Profiler to check if there are any bottlenecks in your code."""

from typing_extensions import override

from lightning.pytorch.profilers.profiler import Profiler


class PassThroughProfiler(Profiler):
    """This class should be used when you don't want the (small) overhead of profiling.

    The Trainer uses this class by default.

    """

    @override
    def start(self, action_name: str) -> None:
        pass

    @override
    def stop(self, action_name: str) -> None:
        pass
