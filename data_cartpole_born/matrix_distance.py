import pennylane as qml
from pennylane import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=[i for i in range(n_qubits+1)])

def ansatz_flatten(state, flat_weights, n_qubits, n_layers=1, change_of_basis=False, entanglement="all2all"):
    #flat_weights = weights.flatten()
    num_weights_per_layer = n_qubits * 2

    if change_of_basis is True:
        for l in range(n_layers):
            for i in range(n_qubits):
                index = l * num_weights_per_layer + i * 2
                qml.Rot(flat_weights[index], flat_weights[index + 1], wires=i)
    else:          
        for l in range(n_layers):
            for i in range(n_qubits):
                index = l * num_weights_per_layer + i * 2
                qml.RZ(flat_weights[index], wires=i)
                qml.RY(flat_weights[index + 1], wires=i)


            if entanglement == "all2all":
                for q1 in range(n_qubits-1):    
                    for q2 in range(q1+1, n_qubits):
                        qml.CNOT(wires=[q1,q2])

            elif entanglement == "mod":
                for q1 in range(n_qubits):
                    qml.CNOT(wires=[q1, (q1+l+1)%n_qubits])

            elif entanglement == "linear":
                for q1 in range(n_qubits-1):    
                    qml.CNOT(wires=[q1, q1+1])

            elif entanglement == "circular":
                for q1 in range(n_qubits):
                    qml.CNOT(wires=[q1, (q1+1)%n_qubits])

            elif entanglement == "nn":
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[1, 2])

            else:
                for q in range(1, n_qubits):
                    qml.CNOT(wires=[q, 0])
                for q in range(2, n_qubits):
                    qml.CNOT(wires=[q, 1])

            if l < n_layers-1:
                qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Y")
                qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Z")


@qml.qnode(dev)
def circuit(s,params):

    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    ansatz_flatten(s, params, n_qubits, n_layers=1, change_of_basis=False, entanglement="all2all")

    for q in range(n_qubits-1):
        #qml.CNOT(wires=[q,n_qubits])
        qml.CNOT(wires=[q,q+1])
        
    return qml.probs(wires=n_qubits-1) 

batch_size = 1000
ss = np.random.random((batch_size,n_qubits))

avg_state = np.mean(ss,axis=0)

n_layers=4
w = np.random.random((n_layers,n_qubits,2)).flatten()

mt_fn = qml.metric_tensor(circuit, approx=None, aux_wire=n_qubits)
qfim_avg_state = 4*mt_fn(np.array(avg_state,requires_grad=False), np.array(w, requires_grad=True))

qfim = np.zeros((len(w),len(w)))    
for s in ss:
    print("starting qfim {}".format(s))
    qfim += 4*mt_fn(np.array(s,requires_grad=False), np.array(w, requires_grad=True))
    print("done qfim {}".format(s))

qfim /= batch_size


#frobenius norm
fn = np.linalg.norm(qfim-qfim_avg_state)
print("distance between matrices qfim and qfim_avg - {}".format(fn))