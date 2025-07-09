import grpc
from . import scheduler_controller_pb2,scheduler_controller_pb2_grpc,workflow_pb2
import cramjam

def package(name,topology):
    # 封装workflow
    workflow = workflow_pb2.Workflow()
    workflow.workflow_name = name
    for node in topology:
        add_node = workflow.topology.add()
        add_node.name = node.name
        add_node.template = node.template
        for depend in node.dependencies:
            add_node.dependencies.append(depend)

    workflow_str = workflow.SerializeToString()
    # 使用snappy压缩
    compressed = cramjam.snappy_compress(workflow_str)

    try:
        # 连接 rpc 服务器
        channel = grpc.insecure_channel('0.0.0.0:6060')
        # 调用 rpc 服务
        stub = scheduler_controller_pb2_grpc.SchedulerControllerStub(channel)
        response = stub.InputWorkflow(scheduler_controller_pb2.InputWorkflowRequest(workflow=[compressed]))
        return response.accept
    except Exception as e:
        print('grpc error')
        return 0

# test
from typing import List
from pydantic import BaseModel
class WorkflowTask(BaseModel):
    id: str = None
    name: str = None
    template: str = None
    phase: str = None
    dependencies: List[str] = []

if __name__ == '__main__':
    topology = []
    topology.append(WorkflowTask(name='A',template='X',dependencies=[]))
    topology.append(WorkflowTask(name='B',template='Y',dependencies=['A']))
    print(package('a',topology))