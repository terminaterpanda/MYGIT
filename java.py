import jpype
from konlpy.tag import Okt

# JVM 경로 설정
jvm_path = "/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home/lib/jli/libjli.dylib"
jpype.startJVM(jvm_path, "-Dfile.encoding=UTF8")
okt = Okt()
