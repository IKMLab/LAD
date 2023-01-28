r"""Fine-tune experiment tools.

Usage:
    import fine_tune

    config = fine_tune.config.TeacherConfig(...)
    dataset = fine_tune.util.load_dataset_by_config(config)
    tokenizer = fine_tune.util.load_teacher_tokenizer_by_config(config)
    model = fine_tune.util.load_teacher_model_by_config(config)
    optimizer = fine_tune.util.load_optimizer_by_config(config, model)
    scheduler = fine_tune.util.load_scheduler_by_config(config, optimizer)

    fine_tune.util.train(
        config,
        dataset,
        model,
        optimizer,
        scheduler,
        tokenizer
    )
    fine_tune.util.evaluation(config, dataset, model, tokenizer)
    fine_tune.util.gen_logits(config, dataset, model, tokenizer)

    config = fine_tune.config.StudentConfig(...)
    tokenizer = fine_tune.util.load_student_tokenizer_by_config(config)
    model = fine_tune.util.load_student_model_by_config(config, tokenizer)
    optimizer = fine_tune.util.load_optimizer_by_config(config, model)
    scheduler = fine_tune.util.load_scheduler_by_config(config, optimizer)

    fine_tune.util.distill(
        config,
        dataset,
        model,
        optimizer,
        scheduler,
        tokenizer
    )
    fine_tune.util.evaluation(config, dataset, model, tokenizer)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# my own modules

import fine_tune.config
import fine_tune.model
import fine_tune.objective
import fine_tune.path
import fine_tune.task
import fine_tune.util
