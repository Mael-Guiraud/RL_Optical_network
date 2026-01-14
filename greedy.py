from line_profiler import profile
import copy
@profile
def update_buffer(buffer):
    for e in buffer[-1]:
        e[0]+=1
    for i in range(len(buffer[-2])):
        e = buffer[-2].pop()
        buffer[-1].append([e[0]+1,e[1],e[2]])
    for i in range(len(buffer)-2,0,-1):
        for _ in range(len(buffer[i-1])):
            a = buffer[i-1].pop()
            buffer[i].append([a[0]+1,a[1],a[2]])
    buffer[0]=[]
    return buffer

@profile
def count_arrivals(a):
    sum = 0
    for i in a:
        sum += len(i)
    return sum

@profile
def find_oldest_message(buffer,fct_calcul_valeur_buffer):
    max = 0
    id_max = 0
    for i,e in enumerate(buffer[-1]):
        if fct_calcul_valeur_buffer(e) >= max:
            max = fct_calcul_valeur_buffer(e)
            id_max =i
    return id_max

@profile
def count_elems_in_buffer(buffer):
    sum = 0
    for e in buffer:
        sum += len(e)
    return sum

@profile
def brick_1_distinct_packets(arrivals,buffer,fct_calcul_valeur_buffer):
    temps_sortie = 0
    max_delay = 0
    sorties =[]
    nb_messages = count_arrivals(arrivals)
    for i in range(len(arrivals)):
        if len(arrivals[i]) == 0 :
            if len(buffer[-1]) != 0:
                elem_sorti = buffer[-1].pop(find_oldest_message(buffer,fct_calcul_valeur_buffer))
                temps_sortie += elem_sorti[0]
                sorties.append([elem_sorti[0],elem_sorti[-1]])
                max_delay = max(max_delay,elem_sorti[0])
        else:
            x = arrivals[i].pop(arrivals[i].index(max(arrivals[i])))
            sorties.append([0,x[-1]])
            for a in arrivals[i]:
                buffer[0].append([0,a[0],a[1]])
        buffer = update_buffer(buffer)
    return max_delay,temps_sortie/nb_messages,sorties

@profile
def min_avg_delay(arrivals,c,B,fct_calcul_valeur_buffer):
    buffer = [ [] for _ in range(c+1)]
    
    brick_1_distinct_packets(copy.deepcopy(arrivals),buffer,fct_calcul_valeur_buffer)
    if count_elems_in_buffer(buffer) > B:
        return -1
    t= brick_1_distinct_packets(copy.deepcopy(arrivals),buffer,fct_calcul_valeur_buffer)
    t= brick_1_distinct_packets(arrivals,buffer,fct_calcul_valeur_buffer)
    return t

@profile
def mrd_buffer(e):
    return (e[0]+e[1][2])/e[1][1]
def average_delay_buffer(e):
    return e[0]+e[1][2]