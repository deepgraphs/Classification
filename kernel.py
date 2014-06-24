#!/usr/bin/env python
__author__ = "VanKos"
__version__ = '$Id$'

from bllipparser import RerankingParser
from bllipparser.ModelFetcher import download_and_install_model
model_dir = download_and_install_model('WSJ', '/tmp/models')
parser = RerankingParser.from_unified_model_dir(model_dir)
import nltk
from nltk.tree import *
from nltk import treetransforms, bracket_parse 
from copy import deepcopy 
import sys
import re
import logging
from os.path import join
from collections import defaultdict      
import math
class TreeNode:
	def __init__(self):
		self.sName=''; # stores the name of the node
		self.pChild=list() # stores the array of pointers to the childern of this node 
		self.nodeID=0
		self.production=''
		self.pre_terminal=0
class nodePair:
	Nx=TreeNode()	               
	Nz=TreeNode()
class OrderedTreeNode:
	def __init__(self):
		self.sName='' 	   # stores the production at the node
		self.node=TreeNode() # stores the pointers to the corresponding node in the tree structure 
class Ntree:
	def __init__(self,rr):
		self.orderedNodeSet=list() 	   # stores the production at the node
		self.root=rr # stores the pointers to the corresponding node in the tree structure 
		self.listSize=0
		self.counter=0
		self.twonorm_PT=1
		self.fillTheNodeList(self.root)
	def fillTheNodeList(self,N):
		# if isinstance(nod,TreeNode):
			# pre_term=1
			# if(len(nod.pChild)>0):
				# pre_term=0
			# map(self.fillTheNodeList,nod.pChild)
			# nod.production=list[*counter].sName=strdup(production);
			# nod.pre_terminal=pre_term
			# self.orderedNodeSet.append(nod)
			# self.counter+=1
			# self.listSize+=1
		if isinstance(N,TreeNode):
			production=''
			if(len(N.pChild)>0):
				production= N.sName + "->" #+ production
				pre_term=1
				for i in range(len(N.pChild)):
					production+= " " + N.pChild[i].sName
					if(len(N.pChild[i].pChild)>0):
						pre_term=0
					self.fillTheNodeList(N.pChild[i]);
				N1=OrderedTreeNode()
				N1.sName=production
				N.production=N1.sName
				N.pre_terminal=pre_term
				N1.node=N
				self.counter+=1
				self.listSize+=1
				self.orderedNodeSet.append(N1)	
class TreeManager:
	def __init__(self):
		self.ID=0
	def buildTreeNode(self,n):
		node= TreeNode()
		node.sName = n
		node.nodeID=self.ID
		return node
	def loadParseTree(self,tree):		
		nod=tree
		if isinstance( nod, nltk.Tree ) :
			root=self.buildTreeNode(str(nod.node))
			# root.production="("+str(nod.node)
			self.ID+=1
			root.pChild=map (self.loadParseTree,nod[0:]) 
		elif isinstance( nod, basestring ) :
			root=self.buildTreeNode(str(nod))
			# root.production=str(str(nod))+")"
			self.ID+=1
		return root
	def print_e(self,a):
		if isinstance(a.sName,basestring):
			sys.stdout.write(a.sName + " ")
	def ppsprint(self,R,way=1):
		if way==1:
			if isinstance(R,TreeNode):
				print map(self.print_e,R.pChild)
				map(self.ppsprint,R.pChild)
			print
		else:
			if isinstance(R,TreeNode):
				for n in R.pChild:
					sys.stdout.write(n.production + " ")
					self.ppsprint(n,2)
class Corpus:
	def __init__(self,trr):
		self.forest=[trr]
class TreeKernel:
	def __init__(self):
		self.test=0
		self.LAMBDA=.4
		self.SIGMA=1 #1 for SST 0 for ST
		self.normalization=3
		self.delta_matrix = defaultdict(dict)
	def destroy_delta(self):
		del self.delta_matrix
		self.delta_matrix = defaultdict(dict)
	def evaluateNorma(self,d):
		for i in range(len(d.forest)):
			intersect=self.determine_sub_lists(d.forest[i],d.forest[i])
			# print "BEFORE"
			# print self.delta_matrix
			[k,intersect,finalvec] =self.evaluateParseTreeKernel(intersect)
			if(k!=0 and (self.normalization == 1 or self.normalization == 3)):
				d.forest[i].twonorm_PT=k; 
			else:
				d.forest[i].twonorm_PT=1;
		return [finalvec , intersect]
	def evaluateParseTreeKernel(self,pairs):
		sum=0
		finalvec=dict()
		for i in range(len(pairs)):
			
			[contr, accumulator]=self.Delta(pairs[i].Nx,pairs[i].Nz)
			# print pairs[i].Nx.production+" : "+pairs[i].Nz.production + " = " + str(contr)
			sum+=float(contr)
			finalvec[pairs[i].Nx.production]=contr
		return [sum,pairs,finalvec]
	def Delta(self,Nx,Nz,accumulator=''):
		prod=1
		# sys.stdout.write( str(Nx.nodeID)+" : "+str(Nz.nodeID))
		try:
			self.delta_matrix[Nx.nodeID][Nz.nodeID]
		except :
			self.delta_matrix[Nx.nodeID][Nz.nodeID]=0
		if(self.delta_matrix[Nx.nodeID][Nz.nodeID] >= 0):
			accumulator += " => "+Nx.sName +" : "+ Nz.sName + "//"+str(self.delta_matrix[Nx.nodeID][Nz.nodeID])
			return [self.delta_matrix[Nx.nodeID][Nz.nodeID],accumulator]#Case 0 (Duffy and Collins 2002);
		else:
			if(Nx.pre_terminal==1 or Nz.pre_terminal==1):
				# print "Case1"
				self.delta_matrix[Nx.nodeID][Nz.nodeID]=self.LAMBDA
				accumulator += " => "+Nx.sName +" : "+ Nz.sName + "//"+str(self.delta_matrix[Nx.nodeID][Nz.nodeID])
				return [self.delta_matrix[Nx.nodeID][Nz.nodeID],accumulator] 
			else:
				# print "Case2"
				for i in range(len(Nx.pChild)):
					if(Nx.pChild[i].production==Nz.pChild[i].production):
						[aar,acc] = self.Delta( Nx.pChild[i], Nz.pChild[i])
						accumulator+=acc
						prod*= self.SIGMA + aar
				self.delta_matrix[Nx.nodeID][Nz.nodeID]=self.LAMBDA*prod
				accumulator += " => "+Nx.sName +" : "+ Nz.sName + "//"+str(self.delta_matrix[Nx.nodeID][Nz.nodeID])
				return [self.delta_matrix[Nx.nodeID][Nz.nodeID],accumulator]
	def determine_sub_lists(self,ONSA,ONSB):
		ONS_a=ONSA.orderedNodeSet
		ONS_b=ONSB.orderedNodeSet
		intersect=list()
		i=0;j=0;
		
		while(i<len(ONS_a) and j<len(ONS_b)):
			if(ONS_a[i].sName>ONS_b[j].sName):
				j+=1
			elif(ONS_a[i].sName<ONS_b[j].sName):
				i+=1
			else:
				j_old=j
				while(i<len(ONS_a) and ONS_a[i].sName==ONS_b[j].sName):
					while(j<len(ONS_b) and ONS_a[i].sName==ONS_b[j].sName):
						keeper=nodePair()
						keeper.Nx=ONS_a[i].node
						keeper.Nz=ONS_b[j].node
						intersect.append(keeper)
						self.delta_matrix[ONS_a[i].node.nodeID][ONS_b[j].node.nodeID]=-1.0;
						j+=1
					i+=1
					j_final=j
					j=j_old
				j=j_final
		return intersect
	def __exit__(self,arg1,arg2,arg3):
		self.destroy_delta()
		del arg1
		del arg2
		del arg3


	def __enter__(self):
		return self
	
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, filename=join('.','Kernel.log'),
						format='%(asctime)s %(levelname)s %(message)s')
	logging.getLogger('').addHandler(logging.StreamHandler())	
	# thelist=['1000','2000']	
	Total_Dict=list()
	Total_Feat=list()
	Labels = list()
	#test_tree="(SBAR (WHADVP how)(S (VP (VBD did)(VP (VB serfdom)(VP (VB develop)(SBAR (IN in)(S (CC and)(S (ADVP then)(VP (VB leave)(NP Russia))))))))(. ?)))"
	manager= TreeManager()
	logging.info("Starting Tree Kernel Calculation...")
	logging.info("Loading vectors into Memory and Parsing.")
	docnum=0
	fil=sys.argv[1]
	# for fil in thelist:
	indx=0
	with open('datasets/train_'+ sys.argv[1] +'.label') as f:
		for line in f:
			# sys.stdout.write(".")
			tree = re.findall('(\w*?):\w*?\s+(.*)',line)
			with  TreeKernel() as kernel:
				# print "Sentence:\n"+tree[0][1]
				ff=parser.parse(tree[0][1])
				# print "Parsed:\n"+str(ff[0].ptb_parse)
				progress= int((float(indx) / float(sys.argv[1]))* 100)#int((indx / int(sys.argv[1]))* 100)
				sys.stdout.write("Progress: %d%%   \r" % (progress) )
				sys.stdout.flush()
				Labels.append(tree[0][0])
				doc=Corpus(Ntree(manager.loadParseTree(nltk.Tree.parse(str(ff[0].ptb_parse)))))
				[vector,intersect]=kernel.evaluateNorma(doc)
				Total_Dict.append(vector)
				indx+=1
				for x in vector:
					if x not in Total_Feat:
						Total_Feat.append(x)
				docnum+=1

	logging.info("Completed Tree Kernel Calculation!")
	logging.info("Total Features : "+ str(len (Total_Feat)))
	logging.info("Total Documents : "+ str(docnum))
	logging.info("Normalizing within unary sphere.")
	indx=0
	for doc in range(len(Total_Dict)):
		Radius=0
		Length=0
		progress= int((float(indx) / float(sys.argv[1]))* 100)#int((indx / int(sys.argv[1]))* 100)
		sys.stdout.write("Progress: %d%%   \r" % (progress) )
		sys.stdout.flush()
		indx+=1
		for feat in Total_Dict[doc]:
			Radius += Total_Dict[doc][feat] * Total_Dict[doc][feat]
		Radius=math.sqrt(Radius)
		for feat in Total_Dict[doc]:
			Total_Dict[doc][feat]=Total_Dict[doc][feat] / Radius
			Length+= Total_Dict[doc][feat] * Total_Dict[doc][feat]
	# logging.info("Outputing to Hard Drive for MatLab.")
	# with open("data/labels","w") as f:
		# f.write("\n".join(Labels))
	# with open("data/features","w") as f:
		# f.write("\n".join(Total_Feat))
	# with open("data/matrix","w") as f:
		# for doc in Total_Dict:
			# for feat in Total_Feat:
				# try:
					# f.write(str(doc[feat]) + "\t")
					# sys.stdout.write(feat + " ")
				# except:
					# f.write('0.0' + "\t")
			# f.write("\n")
	logging.info("Outputing to Hard Drive for PyML.")
	LabelSpace=list(set(Labels))

	with open("data/LabelsPyML","w") as f:
		f.write("\n".join(LabelSpace))
	with open("data/featuresPyML","w") as f:
		f.write("\n".join(Total_Feat))
	index=0
	indx=0
	with open("data/matrixPyML","w") as f:
		for doc in Total_Dict:
			progress= int((float(indx) / float(sys.argv[1]))* 100)#int((indx / int(sys.argv[1]))* 100)
			sys.stdout.write("Progress: %d%%   \r" % (progress) )
			sys.stdout.flush()
			indx+=1
			# if Labels[index].strip() == "ENTY":
				# f.write('1')
			# else :
				# f.write('-1')
			f.write(str(LabelSpace.index(Labels[index])))
			# for feat in doc:
				# try:
					# f.write(" "+ feat.replace(' ','_') + ":" +str(doc[feat]))
					# sys.stdout.write(feat + " ")
				# except:
					# f.write(',0')
			for feat in Total_Feat:
				try:
					f.write("," +str(doc[feat]))
					# sys.stdout.write(feat + " ")
				except:
					f.write(',0')
			f.write("\n")
			index+=1