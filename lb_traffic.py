import matplotlib.pyplot as plt
import numpy as np
import random
import json
import copy


#initialize the matrixes of traffic in a output_link
def init_link_matrix(nb_routes,period):
    return np.zeros((nb_routes,period))
#initialize the matrixes of traffic for all nodes
def init_all_node_matrix(instance,period,nb_routes):
    instance_matrix = []
    for level in instance:
        level_matrix = []
        for node in level:
            node_matrix = []
            for link in node:
                node_matrix.append(init_link_matrix(nb_routes,period))
            level_matrix.append(node_matrix)
        instance_matrix.append(level_matrix)
    return instance_matrix

#initialize the weight of traffic for all arcs
def init_weight_matrix(instance,period):
    instance_matrix = []
    for level in instance:
        level_matrix = []
        for node in level:
            node_matrix = []
            for link in node:
                node_matrix.append(int(random.randint(1,period-1)))
            level_matrix.append(node_matrix)
        instance_matrix.append(level_matrix)
    return instance_matrix
def init_empty_weight_matrix(instance,period):
    instance_matrix = []
    for level in instance:
        level_matrix = []
        for node in level:
            node_matrix = []
            for link in node:
                node_matrix.append(0)
            level_matrix.append(node_matrix)
        instance_matrix.append(level_matrix)
    return instance_matrix
#initialize the remaining tics for traffic for all arcs
def init_remaining_tics_matrix(instance,period):
    instance_matrix = []
    for level in instance:
        level_matrix = []
        for node in level:
            node_matrix = []
            for link in node:
                node_matrix.append(int(period))
            level_matrix.append(node_matrix)
        instance_matrix.append(level_matrix)
    return instance_matrix
def init_non_scheduled_routes(instance,period):
    instance_matrix = []
    for level in instance:
        level_matrix = []
        for node in level:
            node_matrix = []
            for link in node:
                node_matrix.append(len(link))
            level_matrix.append(node_matrix)
        instance_matrix.append(level_matrix)
    return instance_matrix

#initialize the weight of traffic for all arcs
def init_buffer_matrix(instance,period):
    instance_matrix = []
    for level in instance:
        level_matrix = []
        for node in level:
            node_matrix = []
            for link in node:
                node_matrix.append([int(0) for _ in range(period)])
            level_matrix.append(node_matrix)
        instance_matrix.append(level_matrix)
    return instance_matrix

#Find the maximal size of cycle to which belong i
def find_maximal_cycle(instance,i):
    maxi = -1
    for level in instance:
        for subset in level:
            for cycle in subset:
                if i in cycle:
                    maxi  = max(maxi,len(cycle))
    return maxi

#init first level for the given route i: 
#Draw a random number of tics between  period//find_maximal_cycle(instance,i) // 2 and period//find_maximal_cycle(instance,i)
#set Fs[0][i][0][k] = 1 if k in random_tics
def init_first_level(instance,Fs,period,route):
    number_of_random_tics = random.randint(max(period//find_maximal_cycle(instance,route)//2,1),period//find_maximal_cycle(instance,route))
    random_tics = random.sample(list(range(period)), number_of_random_tics)
    for k in random_tics:
        Fs[0][route][0][route][k] = 1
    return Fs

def update_remaining_tics(remaining_tics,instance,route,nb_tics):
    for i,level in enumerate(instance):
        for j,router in enumerate(level):
            for k,cycle in enumerate(router):
                if route in cycle:
                    remaining_tics[i][j][k] -= nb_tics
    return remaining_tics
def update_non_scheduled_routes(non_scheduled_routes,instance,route):
    for i,level in enumerate(instance):
        for j,router in enumerate(level):
            for k,cycle in enumerate(router):
                if route in cycle:
                    non_scheduled_routes[i][j][k] -= 1
    return non_scheduled_routes
def init_first_level_loaded(instance,Fs,remaining_tics,non_scheduled_routes,period,route):
    max_tics,l,r,c = find_minimal_remaining_tics(instance,remaining_tics,non_scheduled_routes,route)
    if max_tics == 0:
        return Fs
    number_of_random_tics = random.randint(max(max_tics//2,1),max_tics)
    random_tics = random.sample(list(range(period)), number_of_random_tics)
    update_remaining_tics(remaining_tics,instance,route,number_of_random_tics)
    update_non_scheduled_routes(non_scheduled_routes,instance,route)
    for k in random_tics:
        Fs[0][route][0][route][k] = 1
    return Fs


#Compute the sum over the columns of a matrix
def sum_columns(matrix):
    return np.sum(matrix,axis=0)


#Given a list of 0 and 1, return the list of eligible index to 0
def eligible_index_to_0(list):
    eligible_index = []
    for i in range(len(list)):
        if list[i] == 0 :
            eligible_index.append(i)
    return eligible_index


def index_in_order(slots,date,min_buff):
    a = []
    if date in slots:
        a.append(date)
    for i in slots:
        if i >= date+min_buff:
            a.append(i)
    for i in slots:
        if i not in a:
            a.append(i)
    return a

def update_buffer_matrix(buffer_matrix,level,id_router,id_cycle,period,first_value,last_value,all_buff):
    if all_buff:
        for i in range(period):
            buffer_matrix[level][id_router][id_cycle][i] += 1
    if first_value < last_value:
        for i in range(first_value,last_value):
            buffer_matrix[level][id_router][id_cycle][i] += 1
    if first_value > last_value:
        for i in range(first_value,period):
            buffer_matrix[level][id_router][id_cycle][i] += 1
        for i in range(0,last_value):
            buffer_matrix[level][id_router][id_cycle][i] += 1


#Given a level, find the id of the router in the level containing route_id
def find_id_router_and_cycle(instance,level,route_id):
    for id,subset in enumerate(instance[level]):
        for id2,cycle in enumerate(subset):
            if route_id in cycle:
                return id,id2
    return -1

#given a route, find the minimal value of remaining tics over the levels
def find_minimal_remaining_tics(instance,remaining_tics,non_scheduled_routes,route_id):
    mini = remaining_tics[0][route_id][0]
    l_min = 0
    r_min = route_id
    c_min = 0

    for level in range(len(instance)):
        router,cycle = find_id_router_and_cycle(instance,level,route_id)
        if remaining_tics[level][router][cycle]//non_scheduled_routes[level][router][cycle] <= mini:
            mini = remaining_tics[level][router][cycle]//non_scheduled_routes[level][router][cycle]
            l_min = level
            r_min = router
            c_min = cycle
    return mini,l_min,r_min,c_min
def schedule_route_at_level(instance,S_matrix,buffer_matrix,weight_matrix,Fs,delays,period,route_id,level,min_buffer_time,lambda_poisson=0.5):
    #init first level 
    max_buffer = 0
    actual_router,actual_cycle=find_id_router_and_cycle(instance,level,route_id)
    #find the id of the router in the level containing the cycle containing route_id
    id_next_router = -1
    id_next_cycle = -1
    for id,subset in enumerate(instance[level+1]):
        for id2,cycle in enumerate(subset):
            if route_id in cycle:
                id_next_router = id
                id_next_cycle = id2
                break
        if id_next_cycle != -1 and id_next_router != -1:
            break
    #For each tics in Fs[level][actual_router][actual_cycle], we schedule it at next level
    for i in range(period):
        if Fs[level][actual_router][actual_cycle][route_id][i] == 1:
            arrival_date_in_next = i+weight_matrix[level][actual_router][actual_cycle]
            #find the eligible slots in the next level
            poisson_value = np.random.poisson(lambda_poisson)
            index = eligible_index_to_0(sum_columns(Fs[level+1][id_next_router][id_next_cycle]))
            prochain_slot = index_in_order(index,arrival_date_in_next%period,min_buffer_time)[min(poisson_value,len(index)-1)]
            max_buffer = max(max_buffer,(prochain_slot-arrival_date_in_next)%period)
            
            Fs[level+1][id_next_router][id_next_cycle][route_id][prochain_slot] = 1
            delai_rajoute = (prochain_slot-arrival_date_in_next)%period
            all_buff = False
            if delai_rajoute < min_buffer_time and delai_rajoute != 0:
                delai_rajoute += period
                all_buff = True
            S_matrix[level+1][id_next_router][id_next_cycle][route_id][arrival_date_in_next%period] = delai_rajoute
            update_buffer_matrix(buffer_matrix,level+1,id_next_router,id_next_cycle,period,arrival_date_in_next%period,prochain_slot,all_buff)
            delays[level+1][id_next_router][id_next_cycle][route_id][prochain_slot] = delai_rajoute + delays[level][actual_router][actual_cycle][route_id][i]
         
    return max_buffer
                

def schedule_all_routes(instance,S_matrix,buffer_matrix,weight_matrix,Fs,delays,remaining_tics,non_scheduled_routes,period,nb_routes,min_buffer_time,lambda_poisson=0.5):
    maxi = 0
    for route_id in range(nb_routes):
        Fs = init_first_level_loaded(instance,Fs,remaining_tics,non_scheduled_routes,period,route_id)
        #Fs = init_first_level(instance,Fs,period,route_id)
    for level in range(len(instance)-1):
        perms = np.random.permutation(nb_routes)
        for route in perms:
            maxi=max(schedule_route_at_level(instance,S_matrix,buffer_matrix,weight_matrix,Fs,delays,period,route,level,min_buffer_time,lambda_poisson),maxi)

    return maxi



#Verifie si toutes les matrices de FS ne contiennent  au plus qu'un seul 1 par colonne
def check_FS(Fs):
    for level in Fs:
        for node in level:
            for link in node:
                if np.sum(link,axis=0).max() > 1:
                    return False
    return True
def check_FS_message(Fs):
    for i,level in enumerate(Fs):
        for j,node in enumerate(level):
            for k,link in enumerate(node):
                if np.sum(link,axis=0).max() > 1:
                    return False,str(np.sum(link,axis=0).max())+"flows at tic "+ str(np.sum(link,axis=0).argmax())+" use : Level "+str(i)+" Node "+str(j)+" Link "+str(k)
    return True, "No contention"
#Verifie si toutes les valeurs de Fs sont 0 ou 1
def check_FS2(Fs):
    for level in Fs:
        for node in level:
            for link in node:
                for i in link:
                    for j in i:
                        if j != 0 and j != 1:
                            return False

    return True
def check_FS2_message(Fs):
    for i,level in enumerate(Fs):
        for j,node in enumerate(level):
            for k,link in enumerate(node):
                for l in link:
                    for m in l:
                        if m != 0 and m != 1:
                            return False,"Value different from 0 or 1 : Level "+str(i)+" Node "+str(j)+" Link "+str(k)+" Value "+str(m)
    return True, "All values are 0 or 1"
#Tous les tics de la route sont bien présents dans la matrice de FS
def check_FS_route(Fs,route):
    nb_1 = np.sum(Fs[0][route][0][route])
    for i,level in enumerate(Fs):
        #compte le nombre de 1 dans la ligne Fs[i][route][link][route]
        for node in range(len(level)):
            for link in range(len(Fs[i][node])):
                v=np.sum(Fs[i][node][link][route])
                if v != 0 and v != nb_1:
                    return False
    return True

#Verifie si 
def check_FS3(Fs,nb_routes):
    for i in range(nb_routes):
        if not check_FS_route(Fs,i):
            return False
    return True
def check_FS_route_message(Fs,route):
    nb_1 = np.sum(Fs[0][route][0][route])
    for i,level in enumerate(Fs):
        #compte le nombre de 1 dans la ligne Fs[i][route][link][route]
        for node in range(len(level)):
            for link in range(len(Fs[i][node])):
                v=np.sum(Fs[i][node][link][route])
                if v != 0 and v != nb_1:
                    return False,(i,node,link,route,v,nb_1)
    return True, "Ok"
def check_FS3_message(Fs,nb_routes):
    for i in range(nb_routes):
        res = check_FS_route_message(Fs,i)
        if not res[0]:
            return False,"Flow "+str(res[1][3])+"sends "+str(res[1][4])+" tics in level "+str(res[1][0])+" in node "+str(res[1][1])+" in link "+str(res[1][2])+" but it send "+str(res[1][5])+" tics in level 0 "
    return True,"Integrity of all flows OK at all levels"
#Vérifie que tous les delais sont bien supérieurs à min_buff
def check_delays(delays,mini_buff):
    for i in delays:
        for j in i:
            for k in j:
                for l in k:
                    for m in l:
                        if m != 0:
                            if m < mini_buff:
                                return False
    return True

#Given two matrices, check if the non zero elements of m1 are also non zero in m2 
def check_zeros_in_matrix(matrix1,matrix2):
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            if matrix1[i][j] == 0 and matrix2[i][j] != 0:
                return False
    return True
#Given Fs and delays, call check matrix on each link of Fs and delays
def check_matrices_Fs_delays(Fs,delays):
    for i in range(len(Fs)):
        for j in range(len(Fs[i])):
            for k in range(len(Fs[i][j])):
                if not check_zeros_in_matrix(Fs[i][j][k],delays[i][j][k]):
                    return False
    return True

#Check if the tics emitted on a link at lvl i arrives well at lvl i+1 at the right time considering delay and weight
def check_if_pattern_ok(S,Fs,weight_matrix,period,instance):
    import copy
    copy_Fs = copy.deepcopy(Fs)
    for i in range(len(Fs)-1):
        for j in range(len(Fs[i])):
            for k in range(len(Fs[i][j])):
                for l in range(len(Fs[i][j][k])):
                    for m in range(len(Fs[i][j][k][l])):
                        if Fs[i][j][k][l][m] == 1:
                            router_id,cycle_id = find_id_router_and_cycle(instance,i,l)
                            next_router_id,next_cycle_id = find_id_router_and_cycle(instance,i+1,l)
                            weight = weight_matrix[i][router_id][cycle_id]
                            if copy_Fs[i+1][next_router_id][next_cycle_id][l][(m+weight+int(S[i+1][next_router_id][next_cycle_id][l][(m+weight)%period]))%period] == 0:
                                return False
                            copy_Fs[i+1][next_router_id][next_cycle_id][l][(m+weight+int(S[i+1][next_router_id][next_cycle_id][l][(m+weight)%period]))%period] = 0
    #Check if all the elements of copy_Fs are 0
    for i in range(1,len(copy_Fs)):
        for j in range(len(copy_Fs[i])):
            for k in range(len(copy_Fs[i][j])):
                for l in range(len(copy_Fs[i][j][k])):
                    for m in range(len(copy_Fs[i][j][k][l])):
                        if copy_Fs[i][j][k][l][m] != 0:
                            return False
    return True

#Check if the tics emitted on a link at lvl i arrives well at lvl i+1 at the right time considering delay and weight
def check_if_pattern_ok_message(S,Fs,weight_matrix,period,instance):
    import copy
    copy_Fs = copy.deepcopy(Fs)
    for i in range(len(Fs)-1):
        for j in range(len(Fs[i])):
            for k in range(len(Fs[i][j])):
                for l in range(len(Fs[i][j][k])):
                    for m in range(len(Fs[i][j][k][l])):
                        if Fs[i][j][k][l][m] == 1:
                            router_id,cycle_id = find_id_router_and_cycle(instance,i,l)
                            next_router_id,next_cycle_id = find_id_router_and_cycle(instance,i+1,l)
                            weight = weight_matrix[i][router_id][cycle_id]
                            if copy_Fs[i+1][next_router_id][next_cycle_id][l][(m+weight+int(S[i+1][next_router_id][next_cycle_id][l][(m+weight)%period]))%period] == 0:
                                return False,"The tic sent from link "+str(l)+" of router "+str(router_id)+" at cycle "+str(cycle_id)+" at level "+str(i)+" at tic "+str(m)+" is not received at link "+str(next_cycle_id)+" of router "+str(next_router_id)+" at level "+str(i+1)+" at tic "+str((m+weight+int(S[i+1][next_router_id][next_cycle_id][l][(m+weight)%period]))%period)
                            copy_Fs[i+1][next_router_id][next_cycle_id][l][(m+weight+int(S[i+1][next_router_id][next_cycle_id][l][(m+weight)%period]))%period] = 0
    #Check if all the elements of copy_Fs are 0
    for i in range(1,len(copy_Fs)):
        for j in range(len(copy_Fs[i])):
            for k in range(len(copy_Fs[i][j])):
                for l in range(len(copy_Fs[i][j][k])):
                    for m in range(len(copy_Fs[i][j][k][l])):
                        if copy_Fs[i][j][k][l][m] != 0:
                            return False,"No matching with tic "+str(m)+" of link "+str(l)+" of router "+str(k)+" at level "+str(i)
    return True,"All tics are well received"

def check_buffer_min(S,c):
    for i in range(len(S)-1):
        for j in range(len(S[i])):
            for k in range(len(S[i][j])):
                for l in range(len(S[i][j][k])):
                    for m in range(len(S[i][j][k][l])):
                        if S[i][j][k][l][m] !=0 and S[i][j][k][l][m] < c:
                            return False
    return True
def check_buffer_min_message(S,Fs,weight_matrix,period,instance,B):
    for i in range(len(S)-1):
        for j in range(len(S[i])):
            for k in range(len(S[i][j])):
                for l in range(len(S[i][j][k])):
                    for m in range(len(S[i][j][k][l])):
                        if S[i][j][k][l][m] !=0 and S[i][j][k][l][m] < c:
                            return False,"There is a delay of "+str(S[i][j][k][l][m])+" at level "+str(i)+" router "+str(j)+" cycle "+str(k)+" route "+str(l)+" tic "+str(m)+" which is less than "+str(c)
    return True,"All delays are greater than "+str(c)

def check_buffer_max(S,Fs,weight_matrix,period,instance,B):
    buffer_matrix = init_buffer_matrix(instance,period)
    for i in range(len(Fs)-1):
        for j in range(len(Fs[i])):
            for k in range(len(Fs[i][j])):
                for l in range(len(Fs[i][j][k])):
                    for m in range(len(Fs[i][j][k][l])):
                        if Fs[i][j][k][l][m] == 1:
                            router_id,cycle_id = find_id_router_and_cycle(instance,i,l)
                            weight = weight_matrix[i][router_id][cycle_id]
                            next_router_id,next_cycle_id = find_id_router_and_cycle(instance,i+1,l)
                            update_buffer_matrix(buffer_matrix,i+1,next_router_id,next_cycle_id,period,(m+weight)%period,(m+weight+int(S[i+1][next_router_id][next_cycle_id][l][(m+weight)%period]))%period,S[i+1][next_router_id][next_cycle_id][l][(m+weight)%period]>period)
    
    if find_max_buffer(buffer_matrix) > B:
        return False
    return True
def check_no_delays_in_first_level(S):
    for i in range(len(S[0])):
        for j in range(len(S[0][i])):
            for k in range(len(S[0][i][j])):
                for l in range(len(S[0][i][j][k])):
                    if S[0][i][j][k][l] != 0:
                        return False,"There are delays at first level at router "+str(i)+" cycle "+str(j)+" route "+str(k)+" tic "+str(l)
    return True,"No delays at first level"


def check_all(S,Fs,weight_matrix,period,instance,B,c):
    #Return True or false and an error message corresponding to the check that didnt pass
    res = check_no_delays_in_first_level(S)
    if not res[0]:
        return False,res[1]
    res= check_FS_message(Fs)
    if not res[0]:
        return False,res[1]
    res= check_FS2_message(Fs)
    if not res[0]:
        return False,res[1]
    res = check_FS3_message(Fs,len(S[0][0][0]))
    if not res[0]:
        return False,res[1]
    res= check_buffer_min_message(S,c)
    if not res[0]:
        return False,res[1]
    res=check_if_pattern_ok_message(S,Fs,weight_matrix,period,instance)
    if not res[0]:
        return False,res[1]
    if not check_buffer_max(S,Fs,weight_matrix,period,instance,B):
        return False,"The maximal buffer is not respected"
    return True,"All checks passed"
    
def compute_Fs(F,S,instance,weight_matrix,period,nb_routes):
    Fs = init_all_node_matrix(instance,period,nb_routes)
    #Recopy F in FS[0]
    for i in range(len(F)):
        for j in range(len(F[i])):
            for k in range(len(F[i][j])):
                for l in range(len(F[i][j][k])):
                    Fs[0][i][j][k][l] = F[i][j][k][l]
    #Compute the other levels from F and S
    for i in range(0,len(Fs)-1):
        for j in range(len(Fs[i])):
            for k in range(len(Fs[i][j])):
                for l in range(len(Fs[i][j][k])):
                    for m in range(len(Fs[i][j][k][l])):
                        if(Fs[i][j][k][l][m] == 1):
                            router_id,cycle_id = find_id_router_and_cycle(instance,i,l)
                            weight = weight_matrix[i][router_id][cycle_id]
                            next_router_id,next_cycle_id = find_id_router_and_cycle(instance,i+1,l)
                            Fs[i+1][next_router_id][next_cycle_id][l][(m+weight+int(S[i+1][next_router_id][next_cycle_id][l][(m+weight)%period]))%period] = 1
    return Fs
def compute_delays(Fs,S,instance,weight_matrix,period,nb_routes):
    delays = init_all_node_matrix(instance,period,nb_routes)
    #Compute the other levels from F and S
    for i in range(0,len(Fs)-1):
        for j in range(len(Fs[i])):
            for k in range(len(Fs[i][j])):
                for l in range(len(Fs[i][j][k])):
                    for m in range(len(Fs[i][j][k][l])):
                        if(Fs[i][j][k][l][m] == 1):
                            router_id,cycle_id = find_id_router_and_cycle(instance,i,l)
                            weight = weight_matrix[i][router_id][cycle_id]
                            next_router_id,next_cycle_id = find_id_router_and_cycle(instance,i+1,l)
                            delays[i+1][next_router_id][next_cycle_id][l][(m+weight+int(S[i+1][next_router_id][next_cycle_id][l][(m+weight)%period]))%period] =  S[i+1][next_router_id][next_cycle_id][l][(m+weight)%period] + delays[i][j][k][l][m]
    return delays

def find_max_buffer(buffer_matrix):
    maxi = buffer_matrix[0][0][0][0]
    for level in buffer_matrix:
        for router in level:
            for cycle in router:
                for slot in cycle:
                        if slot > maxi:
                            maxi = slot
    return maxi
def convert_instance_to_int(instance):
    for i,level in enumerate(instance):
        for j,router in enumerate(level):
            for k,cycle in enumerate(router):
                for l,_ in enumerate(cycle):
                    instance[i][j][k][l] = int(instance[i][j][k][l])

def convert_array_to_list(tab):
    for i,level in enumerate(tab):
        for j,router in enumerate(level):
            for k,cycle in enumerate(router):
                tab[i][j][k] = tab[i][j][k].tolist()

def append_to_json(file_name, data_to_append):
    try:
        # Ouvrir le fichier en mode lecture
        with open(file_name, 'r') as infile:
            # Charger le contenu JSON existant
            existing_data = json.load(infile)
    except FileNotFoundError:
        # Si le fichier n'existe pas encore, initialiser avec une liste vide
        existing_data = {}

    # Ajouter le nouveau dictionnaire à la liste existante
    existing_data[str(len(existing_data))]=data_to_append

    # Écrire la liste mise à jour dans le fichier JSON
    with open(file_name, 'w') as outfile:
        json.dump(existing_data, outfile, indent=2) 

def save_at_json(instance,S_matrix,buffer_matrix,weight_matrix,Fs,delays,period,nb_routes,mini_buff,filename):

    convert_array_to_list(Fs)
    convert_array_to_list(delays)
    convert_array_to_list(S_matrix)
    convert_instance_to_int(instance)
    data = {}
    data['period'] = int(period)
    data['nb_routes'] = int(nb_routes)
    data['nb_levels'] = int(len(instance))
    data['min_buffer_time'] = int(mini_buff)
    data['max_tics_in_buffer'] = int(find_max_buffer(buffer_matrix))
    data['instance'] = instance
    data['weight_matrix'] = weight_matrix
    data['F'] = Fs[0]
    data['S'] = S_matrix
    #data['Fs'] = Fs
    #data['delays'] = delays  
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=2) 



def generate_random_traffic(instance,period,nb_routes,mini_buff,filename): 
    instance = copy.deepcopy(instance)
    instance.insert(0,[[[int(i)]] for i in range(nb_routes)])
    Fs = init_all_node_matrix(instance,period,nb_routes)
    S_matrix = init_all_node_matrix(instance,period,nb_routes)
    delays = init_all_node_matrix(instance,period,nb_routes)
    weight_matrix = init_weight_matrix(instance,period)
    buffer_matrix =init_buffer_matrix(instance,period)
    remaining_tics = init_remaining_tics_matrix(instance,period)
    non_scheduled_routes = init_non_scheduled_routes(instance,period)
    schedule_all_routes(instance,S_matrix,buffer_matrix,weight_matrix,Fs,delays,remaining_tics,non_scheduled_routes,period,nb_routes,mini_buff,0.5)
    if check_FS(Fs) and check_FS2(Fs) and check_FS3(Fs,nb_routes) and check_delays(delays,mini_buff):
        save_at_json(instance,S_matrix,buffer_matrix,weight_matrix,Fs,delays,period,nb_routes,mini_buff,filename)
        return True
    print("Error in the generation of the traffic")
    return 0


def generate_random_traffic_no_weight(instance,period,nb_routes,mini_buff,filename): 
    instance = copy.deepcopy(instance)
    instance.insert(0,[[[int(i)]] for i in range(nb_routes)])
    Fs = init_all_node_matrix(instance,period,nb_routes)
    S_matrix = init_all_node_matrix(instance,period,nb_routes)
    delays = init_all_node_matrix(instance,period,nb_routes)
    weight_matrix = init_empty_weight_matrix(instance,period)
    buffer_matrix =init_buffer_matrix(instance,period)
    remaining_tics = init_remaining_tics_matrix(instance,period)
    non_scheduled_routes = init_non_scheduled_routes(instance,period)
    schedule_all_routes(instance,S_matrix,buffer_matrix,weight_matrix,Fs,delays,remaining_tics,non_scheduled_routes,period,nb_routes,mini_buff,0.5)
    if check_FS(Fs) and check_FS2(Fs) and check_FS3(Fs,nb_routes) and check_delays(delays,mini_buff):
        save_at_json(instance,S_matrix,buffer_matrix,weight_matrix,Fs,delays,period,nb_routes,mini_buff,filename)
        return True
    print("Error in the generation of the traffic")
    return 0


#Calcul la valeur max dans la matrice delays
def max_delay(delays):
    max = 0
    for i in range(len(delays)):
        for j in range(len(delays[i])):
            for k in range(len(delays[i][j])):
                for l in range(len(delays[i][j][k])):
                    for m in range(len(delays[i][j][k][l])):
                        if delays[i][j][k][l][m] > max:
                            max = delays[i][j][k][l][m]
    return max

#Calcul la somme des délais du dernier niveau
def sum_delay(delays):
    sum = 0
    for j in range(len(delays[-1])):
        for k in range(len(delays[-1][j])):
            for l in range(len(delays[-1][j][k])):
                for m in range(len(delays[-1][j][k][l])):
                    sum += delays[-1][j][k][l][m]
    return sum
def avg_delay(Fs,delays):
    avg = 0
    nb_mess = 0
    for j in range(len(delays[-1])):
        for k in range(len(delays[-1][j])):
            for l in range(len(delays[-1][j][k])):
                for m in range(len(delays[-1][j][k][l])):
                    if Fs[-1][j][k][l][m] == 1:
                        nb_mess+=1
                    avg += delays[-1][j][k][l][m]

    return avg/nb_mess
def route_length(route_id,weight_matrix,instance):
    w = 0
    for i,level in enumerate(instance):
        for j,router in enumerate(level):
            for k,cycle in enumerate(router):
                if route_id in cycle:
                    w += weight_matrix[i][j][k]
    return w
    
def max_ratio_delay(delays,weight_matrix,instance):
    maxi = 0
    for j in range(len(delays[-1])):#pour chaque routeur du Dernier level
        for k in range(len(delays[-1][j])):#pour chaque cycle du routeur
            for l in range(len(delays[-1][j][k])):#pour chaque route du cycle
                maxim = 0
                r_len = route_length(l,weight_matrix,instance)
                for m in range(len(delays[-1][j][k][l])):#pour chaque tic de la route
                    val = (delays[-1][j][k][l][m])/r_len
                    maxim = max(maxim,val) 
                maxi = max(maxi,maxim) 
    #print(maxi)
    return maxi
def generate_random_traffic_no_save(instance,period,nb_routes,mini_buff): 
    instance.insert(0,[[[int(i)]] for i in range(nb_routes)])
    Fs = init_all_node_matrix(instance,period,nb_routes)
    S_matrix = init_all_node_matrix(instance,period,nb_routes)
    delays = init_all_node_matrix(instance,period,nb_routes)
    weight_matrix = init_weight_matrix(instance,period)
    buffer_matrix =init_buffer_matrix(instance,period)
    remaining_tics = init_remaining_tics_matrix(instance,period)
    non_scheduled_routes = init_non_scheduled_routes(instance,period)
    schedule_all_routes(instance,S_matrix,buffer_matrix,weight_matrix,Fs,delays,remaining_tics,non_scheduled_routes,period,nb_routes,mini_buff,0.5)
    if check_FS(Fs) and check_FS2(Fs) and check_FS3(Fs,nb_routes) and check_delays(delays,mini_buff):
        return Fs,weight_matrix,delays,sum_delay(delays),max_ratio_delay(delays,weight_matrix,instance),int(find_max_buffer(buffer_matrix))
    print("Error in the generation of the traffic")
    return None
def find_max_load_arc(instance):
    maxi = 0
    max_i=0
    max_j=0
    max_k=0
    for i,level in enumerate(instance):
        for j,router in enumerate(level):
            for k,cycle in enumerate(router):
                if len(cycle) > maxi:
                    maxi = len(cycle)
                    max_i = i
                    max_j = j
                    max_k = k


    return maxi,max_i,max_j,max_k


def count_ones_in_most_loaded_matrix(Fs):
    """
    Compte le nombre de 1 dans la matrice la plus chargée du réseau.

    Args:
        Fs: Les matrices de trafic du réseau
        instance: La structure du réseau

    Returns:
        Tuple contenant (nombre de 1 dans la matrice la plus chargée, niveau, routeur, cycle)
    """
    max_ones = 0
    max_location = (0, 0, 0)  # (level, router, cycle)

    for i, level in enumerate(Fs):
        for j, node in enumerate(level):
            for k, link in enumerate(node):
                # Compter le nombre total de 1 dans cette matrice
                ones_count = np.sum(link)

                if ones_count > max_ones:
                    max_ones = ones_count
                    max_location = (i, j, k)
    # print("Max ones in matrix:", max_ones)
    # print("Location of the matrix:", max_location)
    return max_ones, max_location


#Calcul le nombre de slots libres dans l'arc le plus chargé des fs
def count_free_slots(instance,Fs):
    _,i,j,k = find_max_load_arc(instance)
    
    return np.count_nonzero(sum_columns(Fs[i][j][k]))

## Voici la liste des check a effectuer avant de sauvegarder le traffic dans un fichier json: 
# check_FS(Fs) 
# check_FS2(Fs) 
# check_FS3(Fs,nb_routes) 
# check_delays(delays,mini_buff):
# check_matrices_Fs_delays(Fs,S)
# check_matrices_Fs_delays(Fs,delays)
# check_if_pattern_ok(S,Fs,weight_matrix,period,instance)
# check_buffer_min(S,c)
#check_buffer_max(S,Fs,weight_matrix,period,instance,B)

# On peut rajouter ces tests pour verifier qu'une instance générée par l'IA est bonne.
# L'IA ne nous donnant ni les FS ni les delays, il faudra les calculer avec les fonctions suivantes:
# compute_Fs(F,S,instance,weight_matrix,period,nb_routes))
# compute_delays(Fs,S,instance,weight_matrix,period,nb_routes))