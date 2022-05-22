from collections import namedtuple
from typing import NamedTuple
import kfp
import kfp.components as comp
import kfp.compiler


def add_random(num : int)-> NamedTuple(
	'Outputs',
	[
		('num', int),
	]):
	from collections import namedtuple

	example_output = namedtuple(
		'Outputs',
		['num'])
	return example_output(num + 3)
	return num + 3

def add_number(num : int)-> NamedTuple(
	'Outputs',
	[
		('num', int),
	]):
	from collections import namedtuple
	example_output = namedtuple(
		'Outputs',
		['num'])
	return example_output(num + 5)

def mult_int(a : int, b : int)-> NamedTuple(
	'Outputs',
	[
		('num', int),
	]):
	from collections import namedtuple
	example_output = namedtuple(
		'Outputs',
		['num'])
	return example_output(a * b)

@kfp.dsl.pipeline(
	name='Add Random ',
	description='Add Random'
)
def main_pipeline(a : int, b : int):
	add_random_comp = comp.create_component_from_func(
		add_random
	)
 
	add_random_task1 = add_random_comp(a)
	add_random_task2 = add_random_comp(b)
	
	mult_int_comp = comp.create_component_from_func(
		mult_int
	)
	# print(add_random_task1.outputs)
	mult_int_task = mult_int_comp(add_random_task1.outputs['num'], add_random_task2.outputs['num'])
	print(mult_int_task.outputs['num'])
 
if __name__ == "__main__":
	kfp.compiler.Compiler().compile(main_pipeline, 'random_pipeline.yaml')
