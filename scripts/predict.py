from _cli import parse
from prc import pipeline

a = parse(pipeline.PREDICT, ['all'])
pipeline.run(pipeline.PREDICT, a.stages, a)
