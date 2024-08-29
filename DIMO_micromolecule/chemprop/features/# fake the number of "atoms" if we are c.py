# fake the number of "atoms" if we are collapsing substructures
self.n_real_atoms = mol.GetNumAtoms()
# Get atom features
self.atomic_nums = []   # 原子序数，例如氢原子的系数为1
for i, atom in enumerate(mol.GetAtoms()):
    self.f_atoms.append(atom_features(atom))
    
    atomicnum = atom.GetAtomicNum()
    self.atomic_nums.append(atomicnum)

self.eles = list(set(self.atomic_nums))
self.eles.sort()
self.n_eles = len(self.eles)
self.n_atoms += len(self.eles)+self.n_real_atoms    #self.eles是根据KG得到的新的原子，相当于做node adding数据增强

self.f_eles = [ele_features(self.eles[i]) for i in range(self.n_eles)]
self.f_atoms += self.f_eles
            
self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]

self.atomic_nums += self.eles

for _ in range(self.n_atoms):
    self.a2b.append([])

# Get bond features
for a1 in range(self.n_atoms):
    for a2 in range(a1 + 1, self.n_atoms):
        if a2 < self.n_real_atoms:
            bond = mol.GetBondBetweenAtoms(a1, a2)

            if bond is None:
                continue

            # f_bond = self.f_atoms[a1] + bond_features(bond)
            f_bond = bond_features(bond)
            
        
        elif a1 < self.n_real_atoms and a2 >= self.n_real_atoms:
            if self.atomic_nums[a1] == self.atomic_nums[a2]:
                ele = self.atomic_nums[a1]
                f_bond = hrc_features(ele)
            else:
                continue
                
        elif a1 >= self.n_real_atoms:
            if (self.atomic_nums[a1],self.atomic_nums[a2]) in rel2emb.keys():
                f_bond = relation_features(self.atomic_nums[a1], self.atomic_nums[a2])
            else:
                continue      

        if args.atom_messages:
            self.f_bonds.append(f_bond)
            self.f_bonds.append(f_bond)
        else:
            self.f_bonds.append(self.f_atoms[a1] + f_bond)
            self.f_bonds.append(self.f_atoms[a2] + f_bond)
            
        # Update index mappings
        b1 = self.n_bonds
        b2 = b1 + 1
        self.a2b[a2].append(b1)  # b1 = a1 --> a2
        self.b2a.append(a1)
        self.a2b[a1].append(b2)  # b2 = a2 --> a1
        self.b2a.append(a2)
        self.b2revb.append(b2)
        self.b2revb.append(b1)
        self.n_bonds += 2
        self.bonds.append(np.array([a1, a2]))