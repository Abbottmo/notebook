import numpy as np
cimport numpy as np
cimport cython
import time
from libc.string cimport memset
from libc.stdlib cimport malloc,free
# ctypedef np.int64_t np_int64_t
# ctypedef np.uint16_t np_uint16_t
# ctypedef np.uint32_t np_uint32_t
ctypedef unsigned char            uint8_t
ctypedef unsigned short int         uint16_t
ctypedef unsigned int             uint32_t
ctypedef int                   int32_t
#ref https://www.zybuluo.com/qqiseeu/note/165996

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef WaterDrop(np.ndarray[np.uint16_t,ndim=2] InputImg,uint32_t num_of_drop=10):
    cdef uint16_t maxValue = 65535   
    cdef uint32_t picHeight = <uint32_t> InputImg.shape[0]
    cdef uint32_t picWidth = <uint32_t> InputImg.shape[1]
    cdef np.ndarray[np.uint16_t,ndim=2] inputCopy = InputImg.copy()
    cdef np.ndarray[np.uint32_t,ndim=2] outputImg = np.zeros((picHeight,picWidth),dtype=np.uint32)
    inputCopy[InputImg<2] = maxValue
    
    cdef uint8_t waterWay = 0,isNext=0
    cdef uint32_t j,i,LocX,LocY,k
    cdef uint16_t minP = maxValue
    cdef uint8_t* Label_P =<uint8_t*> malloc(picHeight*picWidth*sizeof(uint8_t))
    if not Label_P:
        raise MemoryError()
    # cdef np.ndarray[np.uint16_t,ndim=2] tmp = np.zeros((3,3),dtype=np.uint16)
    # cdef np.ndarray[np.uint16_t,ndim=1] tmp1 = np.zeros((9,),dtype=np.uint16)
    for j in range(5,picHeight,5):
        for i in range(5,picWidth,5):
            if inputCopy[j,i] == maxValue: continue
            LocX,LocY = i,j
            memset(Label_P,0,picHeight*picWidth*sizeof(uint8_t))
            for k in range(1,26):
                if LocX>picWidth-1 or LocX<2 or LocY>picHeight-1 or LocY<2: break
                minP = inputCopy[LocY,LocX]
                waterWay = 4 #中间为最小
                if minP >= inputCopy[LocY-1,LocX-1]:
                   minP =  inputCopy[LocY-1,LocX-1]
                   waterWay = 0
                if minP >= inputCopy[LocY-1,LocX]:
                   minP =  inputCopy[LocY-1,LocX]
                   waterWay = 1
                if minP >= inputCopy[LocY-1,LocX+1]:
                   minP =  inputCopy[LocY-1,LocX+1]
                   waterWay = 2
                if minP >= inputCopy[LocY,LocX-1]:
                   minP =  inputCopy[LocY,LocX-1]
                   waterWay = 3
                if minP >= inputCopy[LocY,LocX+1]:
                   minP =  inputCopy[LocY,LocX+1]
                   waterWay = 5
                if minP >= inputCopy[LocY+1,LocX-1]:
                   minP =  inputCopy[LocY+1,LocX-1]
                   waterWay = 6
                if minP >= inputCopy[LocY+1,LocX]:
                   minP =  inputCopy[LocY+1,LocX]
                   waterWay = 7
                if minP >= inputCopy[LocY+1,LocX+1]:
                   minP =  inputCopy[LocY+1,LocX+1]
                   waterWay = 8
                # tmp  = inputCopy[LocY-1:LocY+2,LocX-1:LocX+2].copy()
                # tmp1 = tmp.reshape(9)
                # waterWay =<uint8_t> np.argmin(tmp1)
                # minP = <uint16_t> tmp1[waterWay]
                isNext = 0
                while isNext==0:
                    if waterWay == 0:
                        LocX-=1
                        LocY-=1
                        if LocX<2 or LocY<2:
                            isNext=1
                            break
                        minP = inputCopy[LocY,LocX]
                        waterWay = 4
                        Label_P[picWidth*LocY+LocX] = k

                        if minP>= inputCopy[LocY+1,LocX-1] and Label_P[picWidth*(LocY+1)+LocX-1]!=k:
                            waterWay = 6
                            minP = inputCopy[LocY+1,LocX-1]   #左下
                        if minP>= inputCopy[LocY-1,LocX+1] and Label_P[picWidth*(LocY-1)+LocX+1]!=k:
                            waterWay = 2
                            minP = inputCopy[LocY-1,LocX+1]   #右上
                        if minP>= inputCopy[LocY,LocX-1] and Label_P[picWidth*LocY+LocX-1]!=k:
                           waterWay = 3
                           minP = inputCopy[LocY,LocX-1]      #左
                        if minP>= inputCopy[LocY-1,LocX] and Label_P[picWidth*(LocY-1)+LocX]!=k:
                           waterWay = 1
                           minP = inputCopy[LocY-1,LocX]      #上
                        if minP>= inputCopy[LocY-1,LocX-1] and Label_P[picWidth*(LocY-1)+LocX-1]!=k:
                           waterWay = 0 
                           minP = inputCopy[LocY-1,LocX-1]    #左上
                        continue
                    if waterWay == 1:
                        LocY -= 1
                        if LocY<2:
                            isNext=1
                            break
                        minP = inputCopy[LocY,LocX]
                        waterWay = 4
                        Label_P[picWidth*LocY+LocX] = k
                        if minP>= inputCopy[LocY-1,LocX-1] and Label_P[picWidth*(LocY-1)+LocX-1]!=k:
                           waterWay = 0 
                           minP = inputCopy[LocY-1,LocX-1]    #左上
                        if minP>= inputCopy[LocY-1,LocX+1] and Label_P[picWidth*(LocY-1)+LocX+1]!=k:
                            waterWay = 2
                            minP = inputCopy[LocY-1,LocX+1]   #右上
                        if minP>= inputCopy[LocY-1,LocX] and Label_P[picWidth*(LocY-1)+LocX]!=k:
                           waterWay = 1
                           minP = inputCopy[LocY-1,LocX]      #上
                  
                    elif waterWay == 2:
                        LocY -= 1
                        LocX += 1
                        if LocY<2 or LocX>picWidth-2:
                            isNext=1
                            break
                        minP = inputCopy[LocY,LocX]
                        waterWay = 4
                        Label_P[picWidth*LocY+LocX] = k
                        if minP>= inputCopy[LocY-1,LocX-1] and Label_P[picWidth*(LocY-1)+LocX-1]!=k:
                           waterWay = 0 
                           minP = inputCopy[LocY-1,LocX-1]    #左上
                        if minP>= inputCopy[LocY+1,LocX+1] and Label_P[picWidth*(LocY+1)+LocX+1]!=k:
                            waterWay = 8
                            minP = inputCopy[LocY+1,LocX+1]   #右下
                        if minP>= inputCopy[LocY-1,LocX] and Label_P[picWidth*(LocY-1)+LocX]!=k:
                           waterWay = 1
                           minP = inputCopy[LocY-1,LocX] 
                        if minP>= inputCopy[LocY,LocX+1] and Label_P[picWidth*LocY+LocX+1]!=k:
                            waterWay = 5
                            minP = inputCopy[LocY,LocX+1]   #右  
                        if minP>= inputCopy[LocY-1,LocX+1] and Label_P[picWidth*(LocY-1)+LocX+1]!=k:
                            waterWay = 2
                            minP = inputCopy[LocY-1,LocX+1]   #右上
                          #上
                     
                    elif waterWay == 3:
                       LocX-=1
                       if LocX<2:
                          isNext=1
                          break
                       minP = inputCopy[LocY,LocX]
                       waterWay = 4
                       Label_P[picWidth*LocY+LocX] = k
                       if minP>= inputCopy[LocY-1,LocX-1] and Label_P[picWidth*(LocY-1)+LocX-1]!=k:
                           waterWay = 0 
                           minP = inputCopy[LocY-1,LocX-1]    #左上
                       if minP>= inputCopy[LocY+1,LocX-1] and Label_P[picWidth*(LocY+1)+LocX-1]!=k:
                            waterWay = 6
                            minP = inputCopy[LocY+1,LocX-1]   #左下
                       if minP>= inputCopy[LocY,LocX-1] and Label_P[picWidth*LocY+LocX-1]!=k:
                           waterWay = 3
                           minP = inputCopy[LocY,LocX-1]      #左
                    elif waterWay == 4:
                        inputCopy[LocY,LocX] += num_of_drop
                        outputImg[LocY,LocX]+=4000
                        isNext = 1
                        break
                    elif waterWay == 5:
                        LocX += 1
                        if LocX > picWidth-2:
                            isNext=1
                            break
                        minP = inputCopy[LocY,LocX]
                        waterWay = 4
                        Label_P[picWidth*LocY+LocX] = k
                        if minP>= inputCopy[LocY-1,LocX+1] and Label_P[picWidth*(LocY-1)+LocX+1]!=k:
                            waterWay = 2
                            minP = inputCopy[LocY-1,LocX+1]   #右上
                        if minP>= inputCopy[LocY+1,LocX+1] and Label_P[picWidth*(LocY+1)+LocX+1]!=k:
                            waterWay = 8
                            minP = inputCopy[LocY+1,LocX+1]   #右下
                        if minP>= inputCopy[LocY,LocX+1] and Label_P[picWidth*LocY+LocX+1]!=k:
                            waterWay = 5
                            minP = inputCopy[LocY,LocX+1]   #右WWW
                    elif waterWay == 6:
                        LocX -= 1
                        LocY += 1
                        if LocX <2 or LocY>picHeight-2:
                            isNext=1
                            break
                        minP = inputCopy[LocY,LocX]
                        waterWay = 4
                        Label_P[picWidth*LocY+LocX] = k

                        if minP>= inputCopy[LocY-1,LocX-1] and Label_P[picWidth*(LocY-1)+LocX-1]!=k:
                           waterWay = 0 
                           minP = inputCopy[LocY-1,LocX-1]    #左上
                        if minP>= inputCopy[LocY+1,LocX+1] and Label_P[picWidth*(LocY+1)+LocX+1]!=k:
                            waterWay = 8
                            minP = inputCopy[LocY+1,LocX+1]   #右下
                        if minP>= inputCopy[LocY,LocX-1] and Label_P[picWidth*LocY+LocX-1]!=k:
                           waterWay = 3
                           minP = inputCopy[LocY,LocX-1]      #左
                        if minP>= inputCopy[LocY+1,LocX] and Label_P[picWidth*(LocY+1)+LocX]!=k:
                           waterWay = 7
                           minP = inputCopy[LocY+1,LocX]      #下
                        if minP>= inputCopy[LocY+1,LocX-1] and Label_P[picWidth*(LocY+1)+LocX-1]!=k:
                           waterWay = 6
                           minP = inputCopy[LocY+1,LocX-1]   #左下
                       
                    elif waterWay == 7:
                        LocY += 1
                        if LocY>picHeight-2:
                            isNext=1
                            break
                        minP = inputCopy[LocY,LocX]
                        waterWay = 4
                        Label_P[picWidth*LocY+LocX] = k
                        if minP>= inputCopy[LocY+1,LocX-1] and Label_P[picWidth*(LocY+1)+LocX-1]!=k:
                           waterWay = 6
                           minP = inputCopy[LocY+1,LocX-1]   #左下
                        if minP>= inputCopy[LocY+1,LocX+1] and Label_P[picWidth*(LocY+1)+LocX+1]!=k:
                            waterWay = 8
                            minP = inputCopy[LocY+1,LocX+1]   #右下
                        if minP>= inputCopy[LocY+1,LocX] and Label_P[picWidth*(LocY+1)+LocX]!=k:
                           waterWay = 7
                           minP = inputCopy[LocY+1,LocX]      #下
                    elif waterWay == 8:
                        LocX += 1
                        LocY += 1
                        if LocX >picWidth-2 or LocY>picHeight-2:
                            isNext=1
                            break
                        minP = inputCopy[LocY,LocX]
                        waterWay = 4
                        Label_P[picWidth*LocY+LocX] = k

                        if minP>= inputCopy[LocY+1,LocX-1] and Label_P[picWidth*(LocY+1)+LocX-1]!=k:
                           waterWay = 6
                           minP = inputCopy[LocY+1,LocX-1]   #左下
                        if minP>= inputCopy[LocY-1,LocX+1] and Label_P[picWidth*(LocY-1)+LocX+1]!=k:
                            waterWay = 2
                            minP = inputCopy[LocY-1,LocX+1]   #右上
                       
                        if minP>= inputCopy[LocY+1,LocX] and Label_P[picWidth*(LocY+1)+LocX]!=k:
                           waterWay = 7
                           minP = inputCopy[LocY+1,LocX]      #下
                        if minP>= inputCopy[LocY,LocX+1] and Label_P[picWidth*LocY+LocX+1]!=k:
                            waterWay = 5
                            minP = inputCopy[LocY,LocX+1]   #右
                        if minP>= inputCopy[LocY+1,LocX+1] and Label_P[picWidth*(LocY+1)+LocX+1]!=k:
                            waterWay = 8
                            minP = inputCopy[LocY+1,LocX+1]   #右下

    
    free(Label_P)
    return outputImg



