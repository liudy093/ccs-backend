

class imageUtil(object):

    def __init__(self):
        self.base = {'apiVersion': 'argoproj.io/v1alpha1', 'kind': 'Workflow', 'metadata': {'generateName': ''}, 'spec': {'entrypoint': 'diamond', 'templates': []}}
        self.baseImage = {'name': '', 'container': {'image': '', 'command': ['echo', 'success']}}
        self.baseDag = {'name': 'diamond', 'dag': {'tasks': []}}

    def submit_workflow(self, workflow_name, task_list):
        self.base['metadata']['generateName'] = workflow_name + '-'
        template_dict = dict()
        template_list = []
        template_all_list = []
        leap = 1
        for task in task_list:
            if not template_dict.__contains__(task.template):
                name = 'workflow' + str(leap)
                leap = leap + 1
                template_dict[task.template] = name
                template_list.append(task)
            task.template = template_dict[task.template]
            task_temp = dict(task)
            if len(task_temp['dependencies']) <1:
                del task_temp['dependencies']
            del task_temp['id']
            template_all_list.append(task_temp)
        self.baseDag['dag']['tasks'] = template_all_list
        image_list = []
        for template in template_dict:
            # self.baseImage['name'] = template.template
            self.baseImage['name'] = template_dict[template]
            self.baseImage['container']['image'] = template
            image_list.append(self.baseImage.copy())
        image_list.append(self.baseDag)
        self.base['spec']['templates'] = image_list
        return self.base


instance = imageUtil()

