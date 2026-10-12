from _cli import parse
from prc import pipeline

a = parse(pipeline.CV, ['all'])
pipeline.run(pipeline.CV, a.stages, a)
