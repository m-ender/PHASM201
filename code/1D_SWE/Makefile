solver_unbalanced: rp1_shallow_roe_with_efix_unbalanced.f90
	f2py -m shallow_roe_with_efix_unbalanced -c $^

solver_leveque: rp1_shallow_roe_with_efix_leveque.f90
	f2py -m shallow_roe_with_efix_leveque -c $^

all: solver_unbalanced solver_leveque
