from pprint import pprint

from kubernetes import client, config
from kubernetes.client.rest import ApiException
import string


class workflowUtil(object):

    status_convert = {
        'Pending':'disable',
        'Succeeded':'success',
        'Running':'running'
    }

    def __init__(self):
        config.load_incluster_config()
        self.api = client.CustomObjectsApi()
        self.core_api = client.CoreV1Api()
        self.group = 'argoproj.io'
        self.version = 'v1alpha1'
        self.namespace = 'workflow'
        self.plural = 'workflows'

    def submit_workflow(self, body):
        # workflow = Parser(file_name)
        try:
            # create the resource
            response = self.api.create_namespaced_custom_object(
                group=self.group,
                version=self.version,
                namespace=self.namespace,
                plural=self.plural,
                body=body
            )
            pprint(response)
            metadata = response['metadata']
            return metadata['name'], metadata['creationTimestamp']
        except ApiException as e:
            print('Exception when calling CustomObjectsApi->get_namespaced_custom_object: %s\n' % e)

    def get_workflow_status(self, workflow_name):
        api_response = self.get_namespaced_custom_object(workflow_name)
        resultList = {}
        nodeList = api_response['status']['nodes']
        workflow_status = api_response['status']['phase']
        for node in nodeList:
            if nodeList[node]['type'] == 'Pod':
                name = nodeList[node]['displayName']
                resultList[name] = {'phase':self.status_convert[nodeList[node]['phase']], 'id': nodeList[node]['id']}
        # return {'workflow_name': workflow_name, 'creationTimestamp': api_response['metadata']['creationTimestamp'],
        #         'status': api_response['status']['phase'], 'nodes': resultList}
        dag_list = api_response['spec']['templates']
        for image in dag_list:
            if 'dag' in image:
                final_dag_list = image['dag']['tasks']
        i=1
        for node in final_dag_list:
            del node['arguments']
            if 'dependencies' not in node:
                node['dependencies'] = []
            if node['name'] in resultList:
                node['phase'] = resultList[node['name']]['phase']
                node['id'] = resultList[node['name']]['id']
                node['node_info'] = self.read_namespaced_pod(node['id'])
            else:
                node['phase'] = 'disable'
                node['id'] = str(i)
                i = i + 1
                node['node_info'] = ''
        return {'name':workflow_name, 'phase':workflow_status, 'topology':final_dag_list}

    def delete_namespaced_custom_object(self, workflow_name):
        try:
            api_response = self.api.delete_namespaced_custom_object(
                self.group, self.version, self.namespace, self.plural, workflow_name, client.V1DeleteOptions())
            pprint(api_response)
        except ApiException as e:
            print('Exception when calling CustomObjectsApi->delete_namespaced_custom_object: %s\n' % e)

    def get_namespaced_custom_object(self, workflow_name):
        try:
            api_response = self.api.get_namespaced_custom_object(
                group=self.group,
                version=self.version,
                namespace=self.namespace,
                plural=self.plural,
                name=workflow_name)
            return api_response
        except ApiException as e:
            print('Exception when calling CustomObjectsApi->get_namespaced_custom_object: %s\n' % e)
            return None

    def patch_namespaced_custom_object(self, workflow_name, is_suspend):
        try:
            wf = self.get_namespaced_custom_object(workflow_name)
            if is_suspend:
                wf['spec']['suspend'] = True
            else:
                wf['spec']['suspend'] = False
            api_response = self.api.patch_namespaced_custom_object(self.group, self.version, self.namespace, self.plural,
                                                                   workflow_name, wf)
            pprint(api_response)
        except ApiException as e:
            print('Exception when calling CustomObjectsApi->patch_namespaced_custom_object_status: %s\n' % e)

    def list_namespaced_custom_object(self):
        try:
            response = self.api.list_namespaced_custom_object(
                self.group,
                self.version,
                self.namespace,
                self.plural
            )
            list = {}
            for node in response['items']:
                list[node['metadata']['name']]=node['status']['phase']
            return list
        except ApiException as e:
            print(e)

    def read_namespaced_pod(self,pod_name):
        try:
            response = self.core_api.read_namespaced_pod(pod_name,self.namespace)
            return response.spec.node_name
        except ApiException as e:
            print('Exception when calling CoreApi->read_namespaced_pod: %s' % e)


# api_response = workflowUtil.get_namespaced_custom_object('test-c8bqb')
# api_response = workflowUtil().get_workflow_status('workflowtestfinal-cmxfg')
# print(api_response)