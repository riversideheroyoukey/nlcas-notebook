from __future__ import annotations

from notebook.models.branch import Branch
from notebook.models.step import Step


class BranchManager:
    """Operations for advancing the single active branch.

    All methods are static and return new objects — no in-place mutation.
    The caller is responsible for inserting returned objects into the session's
    branch/step dicts.
    """

    @staticmethod
    def advance_head(branch: Branch, step: Step) -> Branch:
        """Return a new Branch with head advanced to the given step.

        Validates that the step belongs to this branch.
        The original Branch object is not modified.

        Raises:
            ValueError: If step.branch_id != branch.branch_id.
        """
        if step.branch_id != branch.branch_id:
            raise ValueError(
                f"Step '{step.step_id}' belongs to branch '{step.branch_id}', "
                f"not '{branch.branch_id}'"
            )
        return branch.model_copy(update={"head_step_id": step.step_id})

