from _cli import parse
from prc import pipeline

a = parse(pipeline.TRAIN, ['all'])
pipeline.run(pipeline.TRAIN, a.stages, a)
