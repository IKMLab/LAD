r"""Configuration for fine-tune and fine-tune distillation experiments.

Usage:
    import fine_tune

    teacher_config = fine_tune.config.TeacherConfig(...)
    teacher_config.save()
    teacher_config = TeacherConfig.load(...)

    student_config = fine_tune.config.StudentConfig(...)
    student_config.save()
    student_config = StudentConfig.load(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# my own modules

from fine_tune.config._base_config import BaseConfig
from fine_tune.config._student_config import StudentConfig
from fine_tune.config._teacher_config import TeacherConfig
from fine_tune.config._gate_config import GateConfig
