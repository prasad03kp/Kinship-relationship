from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
import shutil
from myapp import predict
import os
# Create your views here.

def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def final(request,userid):
    context={'a':1}
    return render(request,'final.html',context)

def predictimage(request):
   
    file_path = os.getcwd()+'\\kinship\\media\\FAM1\\PER1'
    file_path1 = os.getcwd()+'\\kinship\\media\\FAM1\\PER2'
    print(file_path)
    fileobj1=request.FILES['image1']
    fileobj2=request.FILES['image2']
    fs=FileSystemStorage()
    shutil.rmtree(file_path)
    shutil.rmtree(file_path1)
    imagename1=fs.save('FAM1//PER1//inp.jpg',fileobj1)
    imagename2=fs.save('FAM1//PER2//inp.jpg',fileobj2)
    imagename1=fs.url(imagename1)
    imagename2=fs.url(imagename2)
    #write models here
    #if image belongs to or not belongs to it will be stored in result
    result=predict.process()
    res=result.split('/')
    print(res)
    context={'imagename1':imagename1,'imagename2':imagename2,'result':res[0],'score':res[1]}
    
    return render(request,'final.html',context)
    