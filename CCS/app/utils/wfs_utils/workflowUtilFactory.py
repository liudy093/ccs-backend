from . import workflowUtil


class WorkflowUtilFactory():

    def __init__(self):
        self.instance = None

    def get_workflow_util(self):
        if self.instance is None:
            self.instance = workflowUtil.workflowUtil()
        return self.instance

instance = WorkflowUtilFactory()