from itertools import combinations, chain
from six.moves import filter as ifilter
from operator import xor
import copy
import math

default_mempages = {
    "4k":int(10240*1024/4), # 10G
    "2M":512*10, # 10G
    "1G":12, # 12G
} # 32G

def generate_numa_topology(sockets=2, numa_nodes=2, cpus = 4, threads=2,
                           default_mempage=default_mempages, mempages={}):
    topology = []
    for s in range(sockets):
        socket = { "id":s, "nodes":[]}
        topology.append(socket)
        for n in range(numa_nodes):
            numa_id = s*numa_nodes + n
            node = { "id":n, "numa_id": numa_id, "pCPUs":[], "parent": socket}
            pages = mempages.get(numa_id) or default_mempage
            node["mempages"] = copy.deepcopy(pages)
            socket["nodes"].append(node)
            cores_per_node = cpus//numa_nodes
            for c in range(cores_per_node):
                pcpu_id = (cpus*threads*s) + n*cores_per_node + c
                cpu = {"processor_id":s,"core_id":c, "pcpu_id":pcpu_id, "threads":[] , "parent": node}
                node["pCPUs"].append(cpu)
                for t in range(threads):
                    thread = {"processor_id":s, "node":n, "numa":numa_id, 
                        "core_id":c, "used":False, "parent": cpu}
                    cpu_id=((cpus*threads*s) + n*cores_per_node  + (cpus*t) + c)
                    thread["cpu_id"] = cpu_id
                    cpu["threads"].append(thread)
                siblings = []
                for thread in cpu["threads"]:
                    siblings.append(thread["cpu_id"])
                for thread in cpu["threads"]:
                    thread["siblings"] = siblings
    return topology

def filter_available_cpus(cpus):
    return ifilter(lambda cpu: cpu["used"]==False, cpus)

def filter_siblings(allocations):
    for allocation in allocations:
        siblings = set()
        for cpu in allocation:
            if cpu["cpu_id"] in siblings:
                break
            siblings.update(cpu["siblings"])
        else:
            yield allocation

def filter_require_free_siblings(threads, thread_dict):
    valid = set()
    for thread in threads:
        siblings = thread["siblings"]
        if thread["cpu_id"] in valid:
            yield thread
        elif all( thread_dict[sibling]["used"] == False for sibling in siblings):
            valid.update(siblings)
            yield thread

def caclulate_effective_request(request):
    # vm_request = {"vCPUs":4,"hw:cpu_policy":"dedicated", "hw:cpu_thread_policy":"isolate"}
    result = copy.deepcopy(request)

    if result.get("hw:cpu_policy") and not result.get("hw:numa_nodes"):
        result["hw:numa_nodes"] = 1
    
    if result.get("hw:mem_page_size","small")  != "small" and not result.get("hw:numa_nodes"):
        result["hw:numa_nodes"] = 1

    
    return result

def can_isolate(allocation):
        parents = {}
        for cpu in allocation:
            parent = cpu["parent"]
            parents[parent["pcpu_id"]] = parent
            # if no thread from the parent CPUs of the allocation are currently in use we can
            # isolate all thread siblings.
        for parent in parents.values():
            for thread in parent["threads"]:
                if thread["used"]:
                    return False
        return True
        

def filter_numa_nodes_count(allocations, vm_request):
    if not vm_request.get("hw:numa_nodes"):
        for allocation in allocations:
            yield allocation
    for allocation in allocations:
        nodes = set()
        for cpu in allocation:
            nodes.add(cpu["numa"])
        if len(nodes) != vm_request.get("hw:numa_nodes"):
            continue
        else:
            yield allocation

def yield_threads_from_numa_nodes(nodes):
    for node in nodes:
        for cpu in node["pCPUs"]:
            for thread in cpu["threads"]:
                yield thread

def yield_numa_nodes_from_topology(topology):
    for socket in topology:
        for node in socket["nodes"]:
            yield node

def yield_threads_from_topology(topology):
    nodes = yield_numa_nodes_from_topology(topology)
    threads = yield_threads_from_numa_nodes(nodes)
    for thread in threads:
        yield thread


#TODO merge with filter unique numa_nodes via a key fuction.
def filter_unique_thread(threads):
    seen = set()
    for thread in threads:
        tid = thread["cpu_id"]
        if tid in seen:
            continue
        else:
            seen.add(tid)
            yield thread

def filter_unique_numa_node(nodes):
    seen = set()
    for node in nodes:
        nid = node["numa_id"]
        if nid in seen:
            continue
        else:
            seen.add(nid)
            yield node

def filter_numa_sets_by_mempages(numa_sets, vm_request):
    memory_mb = vm_request["memory_mb"]
    pagesize = vm_request["hw:mem_page_size"]
    requested_nodes = vm_request["hw:numa_nodes"]
    pagesize_multiplier =  1024/4 if pagesize == "4k" else 1/2 if pagesize == "2M" else 1/1024
    pages = memory_mb * pagesize_multiplier
    ram_per_node = int(pages // requested_nodes)
    if pagesize == "1G":
        pass
    for numa_set in numa_sets:
        for node in numa_set:
            if node["mempages"][pagesize] < ram_per_node:
                break
        else:
            yield numa_set

def yield_threads_from_numa_sets(numa_sets):
    for numa_set in numa_sets:
        iterators = [yield_threads_from_numa_nodes([node]) for node in numa_set]
        try:
            while True:
                for it in iterators:
                    yield next(it)
        except StopIteration:
            pass

def filter_unique_siblings(threads):
    seen = set()
    for thread in threads:
        tid = thread["cpu_id"]
        if tid in seen:
            continue
        else:
            seen.update(thread["siblings"])
            yield thread


def can_require_siblings(allocation,vm_request, host_threads_per_cpu):
        vcpus = vm_request["vCPUs"]
        requested_nodes = vm_request.get("hw:numa_nodes",1)
        cpus_per_node =math.ceil(vcpus/requested_nodes)
        expected_count = requested_nodes
        if host_threads_per_cpu < cpus_per_node:
            expected_count = math.ceil(vcpus / host_threads_per_cpu)
        partents = {cpu["parent"]["pcpu_id"] for cpu in allocation}
        if  cpus_per_node % host_threads_per_cpu == 0:
            return len(partents) == expected_count
        elif math.ceil(vcpus/requested_nodes) < host_threads_per_cpu:
            return len(partents) <= expected_count
        else:
            return len(partents) <= expected_count + requested_nodes



def group_threads_by_required_siblings(threads, vm_request, host_threads_per_cpu):
    nodes = {}
    vcpus = vm_request["vCPUs"]
    requested_nodes = vm_request.get("hw:numa_nodes",1)
    cpus_per_node =math.ceil(vcpus/requested_nodes)
    for thread in threads:
        nodes.setdefault(thread["numa"],[]).append(thread)
    requested_nodes = vm_request["hw:numa_nodes"]
    while nodes:
        current_nodes = []
        for _ in range(requested_nodes):
            node = nodes.popitem()[1]
            current_nodes.append(node)
        try:
            while True:
                for node_threads in current_nodes:
                    for _ in range(cpus_per_node):
                        yield node_threads.pop()
                    if cpus_per_node % host_threads_per_cpu != 0:
                        for _ in range(host_threads_per_cpu -
                            (cpus_per_node % host_threads_per_cpu)):
                            node_threads.pop()
        except IndexError:
            pass

def claim_cpus(cpus, claim_siblings=False):
    if not claim_siblings:
        for cpu in cpus:
            cpu["used"]=True
    else:
        parents = {}
        for cpu in cpus:
            parent = cpu["parent"]
            parents[parent["pcpu_id"]] = parent
        for parent in parents.values():
            claim_cpus(parent["threads"])

def claim_mempages(cpus, vm_request):
    nodes = {}
    memory_mb = vm_request["memory_mb"]
    pagesize = vm_request["hw:mem_page_size"]
    requested_nodes = vm_request["hw:numa_nodes"]
    pagesize_multiplier =  1024/4 if pagesize == "4k" else 1/2 if pagesize == "2M" else 1/1024
    pages = memory_mb * pagesize_multiplier
    pages_per_node = int(pages // requested_nodes)

    for cpu in cpus:
        numa_id = cpu["numa"]
        if numa_id  not in nodes:
            nodes[numa_id] = cpu["parent"]["parent"]
    mempage_allocation = {}
    for node in nodes.values():
        node["mempages"][pagesize] -= pages_per_node
        mempage_allocation[node["numa_id"]] = {"node":node, "pagesize":pagesize,
            "count":pages_per_node}
    return mempage_allocation

def allocate(vm_request, topology,thread_dict):
    # first create a generator of all free cpus
    all_numa_nodes = yield_numa_nodes_from_topology(topology)
    all_threads = yield_threads_from_numa_nodes(all_numa_nodes)

    # determin if the vm requested a numa topology
    requsted_numa_nodes = vm_request.get("hw:numa_nodes")
    if requsted_numa_nodes:
        # if it did, calulate the combinations of numa nodes
        # the cpus could be allocated from
        numa_sets = combinations(all_numa_nodes,requsted_numa_nodes)
        numa_sets = filter_numa_sets_by_mempages(numa_sets,vm_request)
        # then yeild the threads in lockstep order striping
        # each suchsessive thread across numa nodes.
        all_threads = yield_threads_from_numa_sets(numa_sets)

    
    # get the unique free threads
    unique_threads = filter_unique_thread(all_threads)
    free_threads = filter_available_cpus(unique_threads)
    isolate =  vm_request.get("hw:cpu_thread_policy") == "isolate"
    if isolate:
        free_threads = filter_unique_siblings(free_threads)
    require =  vm_request.get("hw:cpu_thread_policy") == "require"
    if require:
        free_threads = filter_require_free_siblings(free_threads, thread_dict)
        free_threads = group_threads_by_required_siblings(free_threads,
            vm_request, host_threads_per_cpu)
        # free_threads = list(free_threads)
    # generate the set of X choose Y combinaiont of possible
    # cpu pinnings
    options = combinations(free_threads, vm_request["vCPUs"])

    # at this point no iterations have happend. all methods
    # invoked are generators so actul computation will happen
    # lazily as the optios are evalated.

    # elimiate all allocations that provide the correct number
    # of numa nodes. Note this should not be needed anymore.
    options = filter_numa_nodes_count(options,vm_request)

    allocation = None
    if isolate:
        # fileter out all allocations that that self
        # overlap with thread siblings.
        # note this step can now be skiped as we filter
        # out siblings before computeing options.
        options = filter_siblings(options)
        allocation = next(options)
        while(allocation):
            if can_isolate(allocation):
                break
            allocation = next(options)
    elif require:
        allocation = next(options)
        while(allocation):
            if can_require_siblings(allocation, vm_request, host_threads_per_cpu):
                break
            allocation = next(options)
    else:
        allocation = next(options)

    allocation_dict = {"cpus":allocation}
    if allocation:
        claim_cpus(allocation, claim_siblings=isolate)
        allocation_dict["mempages"] = claim_mempages(allocation, vm_request)
    return allocation_dict


# emulate a very large system with 512 threads to make this proablem harder.
host_sockets = 4
host_numa_nodes_per_socket = 2
host_cpus_per_socket = 64
host_threads_per_cpu = 8

topology = generate_numa_topology(sockets=host_sockets, numa_nodes=host_numa_nodes_per_socket,
    cpus = host_cpus_per_socket, threads=host_threads_per_cpu)

thread_dict = { thread["cpu_id"]:thread for thread in yield_threads_from_topology(topology)}
# vm with 4 cores spead on 4 numa nodes explcitly
vm_request = {"vCPUs":4,"hw:cpu_policy":"dedicated",
              "hw:cpu_thread_policy":"isolate", "hw:numa_nodes":4,
              "memory_mb":40960, "hw:mem_page_size": "1G"}

def allocate_for_instance(vm_request, topology):
    effective_request = caclulate_effective_request(vm_request)
    allocation = allocate(effective_request,topology,thread_dict)
    if allocation:
        mapping = ",".join( "%s:%s" % (vcpu,pcpu["cpu_id"]) for vcpu, pcpu in  enumerate(allocation["cpus"]))
        print(mapping)
    else:
        print("pinning failed")
    return allocation

#allocation= allocate_for_instance(vm_request,  topology)

# this is a vm with 4 cores an an implcit virtual numa topology of 1 numa node
vm_request = {"vCPUs":4,"hw:cpu_policy":"dedicated",
              "hw:cpu_thread_policy":"isolate",
              "memory_mb":1024, "hw:mem_page_size": "1G"}
#allocation = allocate_for_instance(vm_request,  topology)


# a 4 core v with 1 implicit numa node and no thread isolation
vm_request = {"vCPUs":4,"hw:cpu_policy":"dedicated",
              "memory_mb":1024, "hw:mem_page_size": "1G"}
#allocation = allocate_for_instance(vm_request,  topology)

# a 8 core vm with 3 numa nodes and isolated threads.
vm_request = {"vCPUs":8,"hw:cpu_policy":"dedicated",
              "hw:cpu_thread_policy":"isolate",
              "hw:numa_nodes":3,
              "memory_mb":1024, "hw:mem_page_size": "4k"}
allocation = allocate_for_instance(vm_request,  topology)

# a 8 core vm with 3 numa nodes and isolated threads.
vm_request = {"vCPUs":8,"hw:cpu_policy":"dedicated",
              "hw:cpu_thread_policy":"require",
              "hw:numa_nodes":2,
              "memory_mb":1024, "hw:mem_page_size": "4k"}
#allocation = allocate_for_instance(vm_request,  topology)

print("bug repoducer")
print("----------------------------------------")

# emulate upstream bug
host_sockets = 1
host_numa_nodes_per_socket = 16
host_cpus_per_socket = 48
host_threads_per_cpu = 2

default_mempages = {
    "1G":61,
} # 32G

topology = generate_numa_topology(
    sockets=host_sockets, numa_nodes=host_numa_nodes_per_socket,
    cpus=host_cpus_per_socket, threads=host_threads_per_cpu,
    default_mempage=default_mempages

)

thread_dict = { thread["cpu_id"]:thread for thread in yield_threads_from_topology(topology)}
# vm with 4 cores spead on 4 numa nodes explcitly
vm_request = {"vCPUs":48,"hw:cpu_policy":"dedicated", 
              "hw:cpu_thread_policy":"prefer", "hw:numa_nodes":8,
              "memory_mb":488*1024, "hw:mem_page_size": "1G"}

allocation = allocate_for_instance(vm_request,  topology)

import pprint
printer = pprint.PrettyPrinter(indent=4, width=150)
# this is big so dont print by default
# printer.pprint(topology)
