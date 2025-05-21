import MDAnalysis as mda

# Load the topology and trajectory
u = mda.Universe("4ake.pdb", "MD_4AKEA_protein.xtc")
# Select atoms present in both topology and trajectory
atoms_in_both = u.select_atoms("all")
# Create a new Universe with the selected atoms
u_new = mda.Universe.empty(len(atoms_in_both), trajectory=True)
u_new.add_TopologyAttr("name", atoms_in_both.names)
u_new.add_TopologyAttr("resnames", atoms_in_both.resnames)
u_new.add_TopologyAttr("resids", atoms_in_both.resids)
u_new.add_TopologyAttr("atoms", atoms_in_both.atoms)
# Transfer coordinates
u_new.load_new(u.trajectory)
# Save the new Universe to a new trajectory file
u_new.trajectory.write("aligned_trajectory.xtc")
