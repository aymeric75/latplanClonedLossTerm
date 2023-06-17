import unified_planning
from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader, PDDLWriter


reader = PDDLReader()
pddl_problem = reader.parse_problem('./domain.pddl', './problem.pddl')
print(pddl_problem)

# Get the compiler from the factory

dcr_kind = pddl_problem.kind

dcr_problem = pddl_problem


# create a version of the "problem" without negative preconditions
with Compiler(
        problem_kind = dcr_kind,
        compilation_kind = CompilationKind.NEGATIVE_CONDITIONS_REMOVING
    ) as negative_conditions_remover:

    # After we have the compiler, we get the compilation result
    ncr_result = negative_conditions_remover.compile(
        dcr_problem,
        CompilationKind.NEGATIVE_CONDITIONS_REMOVING
    )
    ncr_problem = ncr_result.problem
    ncr_kind = ncr_problem.kind

    #assert not ncr_kind.has_negative_conditions()


    w = PDDLWriter(ncr_problem)
    w.write_domain('./domainWithoutNeg.pddl')
    w.write_problem('./problemWithoutNeg.pddl')



    # with OneshotPlanner(name='pyperplan') as planner:
    #     result = planner.solve(ncr_problem)
    #     if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
    #         print("Pyperplan returned: %s" % result.plan)
    #     else:
    #         print("No plan found.")