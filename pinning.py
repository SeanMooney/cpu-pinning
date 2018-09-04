from itertools import combinations, chain
from six.moves import filter as ifilter
from operator import xor
import copy

def generate_numa_topology(sockets=2, numa_nodes=2, cpus = 4, threads=2):
    topology = list()
    for s in range(sockets):
        socket = { "id":s, "nodes":[]}
        topology.append(socket)
        for n in range(numa_nodes):
            numa_id = s*numa_nodes + n
            node = { "id":n, "numa_id": numa_id, "pCPUs":[], "parent": socket}
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

def yield_threads_from_numa_sets(numa_sets):
    for numa_set in numa_sets:
        iterators = [yield_threads_from_numa_nodes([node]) for node in numa_set]
        try:
            while True:
                for it in iterators:
                    yield next(it)
        except StopIteration:
            pass

def allocate_cpus(vm_request, topology):
    # first create a generator of all free cpus
    all_numa_nodes = yield_numa_nodes_from_topology(topology)
    all_threads = yield_threads_from_numa_nodes(all_numa_nodes)

    # determin if the vm requested a numa topology
    requsted_numa_nodes = vm_request.get("hw:numa_nodes")
    if requsted_numa_nodes:
        # if it did, calulate the combinations of numa nodes
        # the cpus could be allocated from
        numa_sets = combinations(all_numa_nodes,requsted_numa_nodes)
        # then yeild the threads in lockstep order striping
        # each suchsessive thread across numa nodes.
        all_threads = yield_threads_from_numa_sets(numa_sets)

    
    # get the unique free threads
    unique_threads = filter_unique_thread(all_threads)
    free_threads = filter_available_cpus(unique_threads)
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
    isolate =  vm_request.get("hw:cpu_thread_policy") == "isolate"
    if isolate:
        # fileter out all allocations that that self
        # overlap with thread siblings.
        options = filter_siblings(options)
        allocation = next(options)
        while(allocation):
            if can_isolate(allocation):
                break
            allocation = next(options)
    else:
        allocation = next(options)

    if allocation:
        claim_cpus(allocation,claim_siblings=isolate)
    return allocation


# emulate dual socket ivybdrive host 2 10Core cpus with cluster on die and hyperthreading on
host_sockets = 2 
host_numa_nodes_per_socket = 2
host_cpus_per_socket = 10
host_threads_per_cpu = 2 

topology = generate_numa_topology(sockets=host_sockets, numa_nodes=host_numa_nodes_per_socket,
    cpus = host_cpus_per_socket, threads=host_threads_per_cpu)

# vm with 4 cores spead on 4 numa nodes explcitly
vm_request = {"vCPUs":4,"hw:cpu_policy":"dedicated", 
              "hw:cpu_thread_policy":"isolate",
              "hw:numa_nodes":4}
effective_request = caclulate_effective_request(vm_request)
allocation = allocate_cpus(effective_request,topology)
if allocation:
    mapping = ",".join( "%s:%s" % (vcpu,pcpu["cpu_id"]) for vcpu, pcpu in  enumerate(allocation))
    print(mapping)
else:
    print("pinning failed")

# this is a vm with 4 cores an an implcit virtual numa topology of 1 numa node
vm_request = {"vCPUs":4,"hw:cpu_policy":"dedicated", 
              "hw:cpu_thread_policy":"isolate"}
effective_request = caclulate_effective_request(vm_request)
allocation = allocate_cpus(effective_request,topology)
if allocation:
    mapping = ",".join( "%s:%s" % (vcpu,pcpu["cpu_id"]) for vcpu, pcpu in  enumerate(allocation))
    print(mapping)
else:
    print("pinning failed")

# a 4 core v with 1 implicit numa node and no thread isolation
vm_request = {"vCPUs":4,"hw:cpu_policy":"dedicated"}
effective_request = caclulate_effective_request(vm_request)
allocation = allocate_cpus(effective_request,topology)
if allocation:
    mapping = ",".join( "%s:%s" % (vcpu,pcpu["cpu_id"]) for vcpu, pcpu in  enumerate(allocation))
    print(mapping)
else:
    print("pinning failed")

# a 8 core vm with 3 numa nodes and isolated threads.
vm_request = {"vCPUs":8,"hw:cpu_policy":"dedicated", 
              "hw:cpu_thread_policy":"isolate",
              "hw:numa_nodes":3}
effective_request = caclulate_effective_request(vm_request)
allocation = allocate_cpus(effective_request,topology)
if allocation:
    mapping = ",".join( "%s:%s" % (vcpu,pcpu["cpu_id"]) for vcpu, pcpu in  enumerate(allocation))
    print(mapping)
else:
    print("pinning failed")


import pprint
printer = pprint.PrettyPrinter(indent=4, width=150)
# this is big so dont print by default
#printer.pprint(topology)