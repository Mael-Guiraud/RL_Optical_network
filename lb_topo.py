import math
import numpy as np
def build_next_cycle(perm, begin):
    cycle = []
    i = begin
    while perm[i] != -1:
        cycle.append(i)
        j = perm[i]
        perm[i] = -1
        i = j
    return cycle

def permutation_to_cycles(perm):
    cycles=[]
    for i in range(len(perm)):
        if perm[i] != -1:
            cycles.append(build_next_cycle(perm,i))
    return cycles

def transform_first_stage(cycles):
    for i in range(len(cycles)):
        cycles[i] = [cycles[i]]
    return cycles
Set = transform_first_stage([[1,2],[3,4]])


def get_element_from_subSet(subSet):
    elements = []
    for cycle in subSet:
        for i in cycle:
            elements.append(i)
    return elements

#Given a set and an element , return the cycle in which the element is
def get_cycle_from_element(Set,element):
    for subSet in Set:
        for cycle in subSet:
            if element in cycle:
                return cycle
    return None

#Given a set and an element, return the subSet in which the element is
def get_subSet_from_element(Set,element):
    for subset in Set:
        if element in get_element_from_subSet(subset):
            return subset
    return None


def merge_subSets(Set,subSet1,subSet2):
    for i in Set[subSet2]:
        Set[subSet1].append(i)
    Set.pop(subSet2)
    return Set

#Given a cycle and a subSet, return True if at least one element of the cycle is in the subSet of cycles
def cycle_intersect_subSet(cycle,subSet):
    for i in cycle:
        if i in get_element_from_subSet(subSet):
            return True
    return False

#Given a cycle and a Set, return a list of index of all subSets for which cycleintersectsubSet is true
def get_id_of_intersection(Set,cycle):
    id = []
    for i in range(len(Set)):
        if cycle_intersect_subSet(cycle,Set[i]):
            id.append(i)
    return id
def create_subSets(Set1,Set2):
    for subSet in Set1:
        for cycle in subSet:
            ids= get_id_of_intersection(Set2,cycle)
            nb_removed_ids = 0
            while len(ids) > 1:
                merge_subSets(Set2,ids[0],ids.pop(1)-nb_removed_ids)
                nb_removed_ids += 1

def check_property_1(level0,level1):
    for subSet in level0:
        for cycle in subSet:
            if len(get_id_of_intersection(level1,cycle)) > 1:
                return False
    return True



def is_valid(level0,level1):
    for subSet in level0:
        ids = []
        for cycle in subSet:
            new = get_id_of_intersection(level1,cycle)
            if len(new) > 1:
                return False
            ids.append(new[0])
        #Check if there is a duplicate element in ids
        if len(ids) != len(set(ids)):
            return False
    return True

def generate_instance(nb_routes,nb_levels):
    nb_fails = [0]*nb_levels
    previous_level = transform_first_stage(permutation_to_cycles(np.random.permutation(nb_routes)))
    """while len(previous_level) <= nb_levels:
        previous_level = transform_first_stage(permutation_to_cycles(np.random.permutation(nb_routes)))
        nb_fails[0] += 1"""
    instance = [previous_level]
    
    for level in range(1,nb_levels):
        new_level = transform_first_stage(permutation_to_cycles(np.random.permutation(nb_routes)))
        create_subSets(previous_level,new_level)
        while not is_valid(previous_level,new_level):
            new_level = transform_first_stage(permutation_to_cycles(np.random.permutation(nb_routes)))
            create_subSets(previous_level,new_level)
            nb_fails[level] += 1
            if nb_fails[level] > math.sqrt(math.factorial(nb_routes)):
                return None,None
        instance.append(new_level)
        previous_level = new_level
    return instance,nb_fails


def check_weakly_coherent(instance,nb_routes):
    weakly_coherent = np.zeros((nb_routes,nb_routes))
    for l,level in enumerate(instance):
        for subset in level:
            for cycle in subset:
                for a,i in enumerate(cycle):
                    for j in range(a+1,len(cycle)):
                        if weakly_coherent[i][cycle[j]] == 1:
                            return False
                    if(l>0):
                        C = get_cycle_from_element(instance[l-1],i)
                        for k in C:
                            if k not in cycle:
                                weakly_coherent[i][k] = 1
                                weakly_coherent[k][i] = 1
    return True

def check_strongly_coherent(instance,nb_routes):
    strongly_coherent = np.zeros((nb_routes,nb_routes))
    for l,level in enumerate(instance):
        for subset in level:
            ids = get_element_from_subSet(subset)
            for a,i in enumerate(ids):
                for j in range(a+1,len(ids)):
                    if strongly_coherent[i][ids[j]] == 1:
                        return False
                    
                if(l>0):
                    X = get_element_from_subSet(get_subSet_from_element(instance[l-1],i))
                    for k in X:
                        if k not in ids:
                            strongly_coherent[i][k] = 1
                            strongly_coherent[k][i] = 1

    return True