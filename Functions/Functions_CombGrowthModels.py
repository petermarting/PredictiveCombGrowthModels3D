# Predictive Comb Growth Models for 3D Honey Bee Nests

# from Marting et al. 2023 Proc. B.

# doi: 10.1098/rspb.2022.2565



# Import required packages
import pandas as pd
import numpy as np
import cv2
import random



def arrange_nest(masks, colonyname, c, w, data):
        
    """ 
    Arange 3d nest into the position and orientation that week (important for shuffled colonies whose nest arrangement is different every week)
    
    Args:
        masks: numpy array containing comb images of nests from the colony and timepoint to be arranged
        colonynames: an array of string names for colonies
        c: an integer specifying the colony to be arranged
        w: an integer specifying the week
        data: a dataframe with information about frame id and frame position for all colonies at all timepoints
    
    Return:
        masksoriented: the nest with frames arranged by their actual position for the specified colony at the specified timepoint, in the form of a 3D numpy array
        
    """
    
    data = data[(data['colony']==colonyname)&(data['week']==w+2)&(data['side']=='a')]  #take a subset of dataframe relivant to the focal colony and week
    data = data.sort_values('position') # sort subset dataframe by frame position
    frameorder = data.beeframe.tolist() # get a list of the order of frame IDs
    inverted = data.invert_frame.tolist() # get a list of which frames have flipped orientations
    masksoriented=[] # initiate empty array to be populated by 2D comb images
    for f in range(len(masks)): # for all the frames in the nest
        mask = masks[int(frameorder[f]-1)] # get the frame from the nest with the frame ID in the next true position (-1 because python starts count at 0, but dataframe ID count starts at 1)
        if inverted[f]=='y': # if the orientation of current frame is flipped according to our inverted list 
            mask = np.flip(mask, 1) # flip the current 2D frame about the vertical axis
        masksoriented.append(mask) # add the current frame to growing list of oriented frames
    masksoriented = np.array(masksoriented) #after loop is done, turn the list of 2D oriented comb images into a 3D numpy array
    
    return masksoriented



def model_dilation(masks, c, w, data, colonynames, xbeespace, ybeespace):
    
    """ 
    This model takes a 3D nest at the begining of a week, calculates how much comb growth occured by the end of the week, and generates a prediction of what the 3D nest will be at the end of the week using a simple iterative dilation function informed by how much wax was deposited 
    
    Args:
        masks: numpy array containing comb images of nests from all colonies and all timepoints
        colonynames: an array of string names for colonies
        c: an integer specifying the focal colony
        w: an integer specifying the focal week
        data: a dataframe with information about frame id and frame position for all colonies at all timepoints
        xbeespace: number of pixels to designate as bee space at the bottom of the frame, so dilation won't enter this space, optional - can be set to 0 (default is 10)
        ybeespace: number of pixels to designate as bee space on either side of the frame, so dilation won't enter this space, optional - can be set to 0 (default is 0)
    
    Return:
        masksweek: 3D numpy array of the nest in the initial state at the begining of the week, with frames arranged in the order at specifyed timepoint
        masksweekobs: 3D numpy array of the nest as observered at the end of the week, with frames arranged in the order at specifyed timepoint
        masksweekpred: 3D numpy array of the nest as predicted by the dilation model at the end of the week, with frames arranged in the order at specifyed timepoint
        accuracy: the percentage of overlapping comb area between predicted and observed nests
        areamatch: a percentage of how closely the total predicted nest area matched the observered nest area (should always be very close to 100%)
    """
    
    masksweek = masks[c][w] # defines the focal nest inital state (beginning of week) as a 3d numpy array based on colony c and week w
    masksweek = arrange_nest(masksweek, colonynames[c], c, w, data) # checks the database on how the frames where positoned in the inial state, and arranges the 3D numpy array of the nest accordingly. This is primarily for visualization purposes since the model does not account for 3D structure.
    
    masksweekobs = masks[c][w+1] # defines the focal nest at the end of the week with w+1
    masksweekobs = arrange_nest(masksweekobs, colonynames[c], c, w, data) # again, arranges the end of week nest according to the positional arrangement of frames from in the inital state 

    if (xbeespace>0) & (ybeespace>0): # removes specified beespace from predicted 3D nest array so comb dilation won't incroach. This empty beespace will be added back later
        masksweekpred = np.copy(masksweek[:,:-ybeespace,xbeespace:-xbeespace])
    elif (xbeespace>0) & (ybeespace==0):
        masksweekpred = np.copy(masksweek[:,:,xbeespace:-xbeespace])
    elif (xbeespace==0) & (ybeespace>0):
        masksweekpred = np.copy(masksweek[:,:-ybeespace,:])
    else:
        masksweekpred = np.copy(masksweek) # create a predicted 3d nest by copying the initail state nest for now (dilation will occur below)

    waxbudget = (np.sum(masksweekobs)-np.sum(masksweek)) # Calulate the wax budget in comb pixels by subtracting total comb in nest at the end of the week by the comb at the begining of the week
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) # create the kernel shape that will be used to dilate the comb. This shape is basically a 3x3 pixel cross with the focal pixel in the center.
    repeat = True # initialize repeate as true before follwing loop
    while repeat:
        for f in range(len(masksweekpred)): # for each frame in the predicted nest
            maskpred = np.copy(masksweekpred[f]) # create a copy of current 2D frame from the predicted 3D nest array
            if np.min(maskpred)==0: #if the frame is not full of comb
                maskprednew = cv2.dilate(maskpred,kernel2,iterations = 1) # dilate the comb by 1 iteration (dilation funcion iterates through all the pixels in the 2D array, when it encounters a comb pixel (value of 2 rather than 0) it turns all other pixels in the kernel to comb that are not already)
                prednew_area = int(np.sum(maskprednew)) #calculate the area (in pixels) of the dilated comb
                pred_area = int(np.sum(maskpred)) #calculate the area of the comb before it was just dilated
                added_area = prednew_area - pred_area #caluculate the newly added comb from the dilation
                waxbudget = waxbudget - added_area #subtract the newly added comb from the wax budget
                if waxbudget > 0: # if there is still wax in the wax budget, add the newly dilated comb frame back to the 3D predicted nest and move onto next frame
                    masksweekpred[f] = np.copy(maskprednew)
                    repeat = True
                else: # if the wax budget is depleted, do not add the newly dilated comb to the prediction and turn repeat to false, ending the dilation phase
                    repeat = False

    if xbeespace+ybeespace>0: #add the empty beespace back to the 3D predicted nest 
        maskstemp = []
        for f in range(len(masksweekpred)):
            maskpred = masksweekpred[f]
            ybeespacemask = np.full((ybeespace, masksweek[f].shape[1]-(xbeespace*2)), 0)
            maskpred = np.concatenate((maskpred, ybeespacemask), axis=0).astype(np.uint8)
            xbeespacemask = np.full((masksweek[f].shape[0], xbeespace), 0)
            maskpred = np.concatenate((xbeespacemask, maskpred), axis=1).astype(np.uint8)
            maskpred = np.concatenate((maskpred, xbeespacemask), axis=1).astype(np.uint8)     
            maskstemp.append(maskpred)
        masksweekpred = np.array(maskstemp)
        
        
    evaldiff_area_ttl = 0 # initialize the variable that will calculate the number of non-overlapping pixels between predicted and observed end-of-week-nests
    obspred_area_ttl = 0
    for f in range(len(masksweek)): # for each frame in the nest
        maskobs_mask = masksweekobs[f] + masksweek[f] # add together the focal frame at the start of week and observed end of week, resulting in a 2D frame that has double the value of comb where comb is present in both start and end of week frames)
        maskobs_mask = np.where(maskobs_mask == 4, 0, maskobs_mask) # remove comb area present in the initial state (start of week)
        maskobs_mask_area = np.sum(maskobs_mask) # the area of observed new growth

        maskpred_mask = masksweekpred[f] + masksweek[f] # repeat process above but with the predicted end of week growth instead of observed
        maskpred_mask = np.where(maskpred_mask == 4, 0, maskpred_mask)
        maskpred_mask_area = np.sum(maskpred_mask)

        evalim = maskobs_mask + maskpred_mask # add frames of new observed growth and new predicted growth together
        evalim = np.where(evalim == 2, 0, evalim)  # wherever there's comb that does NOT overlap, delete it
        evaldiff_area = int(np.sum(evalim)) # sum the area where the comb does overlap (needs to be double here to accound for both observed and predicted frames, see below obspred_area_ttl)
                
        evaldiff_area_ttl = evaldiff_area_ttl + evaldiff_area # add the focal frame overlap area bewtween observed and predicted to the running nest total overlap
        obspred_area_ttl = obspred_area_ttl + (maskobs_mask_area + maskpred_mask_area) # add the total new growth comb area of both observed and predicted (not just overlap but all new comb) to the running nest total new growth
        
    accuracy = (evaldiff_area_ttl/obspred_area_ttl)*100 # calculate the percentage of the total new oservered nest area that overlaps with the predicted nest area
    areamatch = round(((int(np.sum(masksweekpred))-int(np.sum(masksweekobs)))/int(np.sum(masksweekobs)))*100, 2) # a check to see how close the total predicted area is to the observed area (should be about 100%)
    
    return masksweek, masksweekobs, masksweekpred, accuracy, areamatch




def model_neighbor(masks, c, w, data, colonynames, second_round=True, downsample=True):
    
    """ 
    This model takes a 3D nest at the begining of a week, calculates how much comb growth occured by the end of the week, and generates a prediction of what the 3D nest will be at the end of the week by adding comb between gaps and on either side of existing comb 
    
    Args:
        masks: numpy array containing comb images of nests from all colonies and all timepoints
        colonynames: an array of string names for colonies
        c: an integer specifying the focal colony
        w: an integer specifying the focal week
        data: a dataframe with information about frame id and frame position for all colonies at all timepoints
        second_round: Boolean value for whether to proceed to the second phase of the model if there is still wax left in the wax budget. The first phase places comb in gaps between existing comb, while the second phase adds comb to either side of existing comb.
        downsample: Boolean value for whether to downsample the original nest array
    
    Return:
        masksweek: 3D numpy array of the nest in the initial state at the begining of the week, with frames arranged in the order at specifyed timepoint
        masksweekobs: 3D numpy array of the nest as observered at the end of the week, with frames arranged in the order at specifyed timepoint
        masksweekpred: 3D numpy array of the nest as predicted by the neighbor model at the end of the week, with frames arranged in the order at specifyed timepoint
        accuracy: the percentage of overlapping comb area between predicted and observed nests
        areamatch: a percentage of how closely the total predicted nest area matched the observered nest area (should always be very close to 100%)
    """    

    masksweek = masks[c][w] # defines the focal nest inital state (beginning of week) as a 3d numpy array based on colony c and week w
    masksweek = arrange_nest(masksweek, colonynames[c], c, w, data) # checks the database on how the frames where positoned in the inial state, and arranges the 3D numpy array of the nest accordingly. This is primarily for visualization purposes since the model does not account for 3D structure.
    
    masksweekobs = masks[c][w+1] # defines the focal nest at the end of the week with w+1
    masksweekobs = arrange_nest(masksweekobs, colonynames[c], c, w, data) # again, arranges the end of week nest according to the positional arrangement of frames from in the inital state 
    
    if downsample: # if downsample is True, reduce the size of the nest arrays by only using every 7th value in the array
        masksweek = masksweek[:,::7,::7] 
        masksweekobs = masksweekobs[:,::7,::7]

    masksweekpred = np.copy(masksweek) # create a predicted 3d nest by copying the initail state nest for now (adding comb will occur below)

    waxbudget = (np.sum(masksweekobs)-np.sum(masksweek))/2 # Calulate the wax budget in comb pixels by subtracting total comb in nest at the end of the week by the comb at the begining of the week (divide by 2 because values for comb in the array are 2, and we just want number of pixelscomb there are)
    cont = True # initialize cont (for continue) as true before follwing loop
    for f in range(len(masksweekpred)): # for each frame in the predicted nest
        f = len(masksweekpred)-1-f # start at the front of the nest (frame 10 is at the entrance)
        for y in range(len(masksweekpred[f])): # from top to bottom of the 2D comb image
            for x in range(len(masksweekpred[f][y])): # from side to side of the 2D comb image (basically checking all pixels in the array)
                if cont: # if cont is still True
                    if masksweekpred[f][y][x]==0: # if focal pixel IS EMPTY
                        neibcount = 0 # its neighbor count starts at 0
                        if f>0: # if we are NOT looking last frame of the box (the one against the box wall)
                            if masksweekpred[f-1][y][x]>0: # check if the there is comb at focal position in one frame forward
                                neibcount = neibcount + 1 # if there is, that add 1 to the neighbor count
                        if f<len(masksweekpred)-1: # if we are NOT looking first frame of the box (the one against the box wall)
                            if masksweekpred[f+1][y][x]>0: # check if the there is comb at focal position in one frame backward
                                neibcount = neibcount + 1 # if there is, that add 1 to the neighbor count

                        if neibcount==2: # if the there is comb on either side of this empty focal comb
                            masksweekpred[f][y][x] = 2 # add a comb there 
                            waxbudget = waxbudget - 1 # remove comb from the wax budget
                            if waxbudget<=0: # if the wax budget is depleted, set cont to False and do not repeat
                                cont = False

    if cont & second_round: # if there is still wax in the wax budget and we have specified to proceed to second phase
        for f in range(len(masksweekpred)): # for each frame in the predicted nest
            f = len(masksweekpred)-1-f # start at the front of the nest (frame 10 is at the entrance)
            for y in range(len(masksweekpred[f])): # from top to bottom of the 2D comb image
                for x in range(len(masksweekpred[f][y])): # from side to side of the 2D comb image (basically checking all pixels in the array)
                    if cont: # if cont is still True
                        if masksweekpred[f][y][x]==2: # if focal pixel HAS COMB
                            if f>0: # if we are NOT looking last frame of the box (the one against the box wall)
                                if masksweekpred[f-1][y][x]==0: # check if the there is comb at focal position in one frame forward
                                    masksweekpred[f-1][y][x] = 2 # if there isn't, add comb there
                                    waxbudget = waxbudget - 1 # remove comb from the wax budget
                                    if waxbudget<=0: # if the wax budget is depleted, set cont to False and do not repeat
                                        cont = False  
                            if f<len(masksweekpred)-1: # if we are NOT looking first frame of the box (the one against the box wall)
                                if masksweekpred[f+1][y][x]==0: # check if the there is comb at focal position in one frame backward
                                    masksweekpred[f+1][y][x] = 2 # if there isn't, add comb there
                                    waxbudget = waxbudget - 1 # remove comb from the wax budget, if the wax budget is depleted, set cont to False and do not repeat
                                    if waxbudget<=0:
                                        cont = False  

                                            
    evaldiff_area = 0 # initialize the variable that will calculate the number of non-overlapping pixels between predicted and observed end-of-week-nests
    obspred_area = 0
    for f in range(len(masksweek)): # for each frame in the nest
        
        maskobs_mask = masksweekobs[f] + masksweek[f] # add together the focal frame at the start of week and observed end of week, resulting in a 2D frame that has double the value of comb where comb is present in both start and end of week frames)
        maskobs_mask = np.where(maskobs_mask == 4, 0, maskobs_mask) # remove comb area present in the initial state (start of week)
        maskobs_mask_area = np.sum(maskobs_mask) # the area of observed new growth

        maskpred_mask = masksweekpred[f] + masksweek[f] # repeat process above but with the predicted end of week growth instead of observed
        maskpred_mask = np.where(maskpred_mask == 4, 0, maskpred_mask)
        maskpred_mask_area = np.sum(maskpred_mask)

        evalim = maskobs_mask + maskpred_mask # add frames of new observed growth and new predicted growth together
        evalim = np.where(evalim == 2, 0, evalim) # wherever there's comb that does NOT overlap, delete it
        evaldiff_area = evaldiff_area + int(np.sum(evalim)) # sum the area where the comb does overlap (needs to be double here to accound for both observed and predicted frames, see below obspred_area_ttl) and add to running total
        obspred_area = obspred_area + (maskobs_mask_area + maskpred_mask_area) # add the total new growth comb area of both observed and predicted (not just overlap but all new comb) to the running nest total new growth
        
    accuracy = (evaldiff_area/obspred_area)*100 # calculate the percentage of the total new oservered nest area that overlaps with the predicted nest area
    areamatch = ((np.sum(masksweekpred)-np.sum(masksweekobs))/np.sum(masksweekobs))*100 # a check to see how close the total predicted area is to the observed area (should be about 100%)
    
    return masksweek, masksweekobs, masksweekpred, accuracy, areamatch



def model_random_placement(masks, c, w, data, colonynames, allsides=False, downsample=True):
    
    """ 
    This model takes a 3D nest at the begining of a week, calculates how much comb growth occured by the end of the week, and generates a prediction of what the 3D nest will be at the end of the week by adding comb randomly to existing comb and tops of frames
    
    Args:
        masks: numpy array containing comb images of nests from all colonies and all timepoints
        colonynames: an array of string names for colonies
        c: an integer specifying the focal colony
        w: an integer specifying the focal week
        allsides: Boolean value for whether to add comb to all sides of frame (top, bottom, left, and right) or just top
        downsample: Boolean value for whether to downsample the original nest array
    
    Return:
        masksweek: 3D numpy array of the nest in the initial state at the begining of the week, with frames arranged in the order at specifyed timepoint
        masksweekobs: 3D numpy array of the nest as observered at the end of the week, with frames arranged in the order at specifyed timepoint
        masksweekpred: 3D numpy array of the nest as predicted by random wax placement at the end of the week, with frames arranged in the order at specifyed timepoint
        accuracy: the percentage of overlapping comb area between predicted and observed nests
        areamatch: a percentage of how closely the total predicted nest area matched the observered nest area (should always be very close to 100%)
    """    

    masksweek = masks[c][w] # defines the focal nest inital state (beginning of week) as a 3d numpy array based on colony c and week w
    masksweek = arrange_nest(masksweek, colonynames[c], c, w, data) # checks the database on how the frames where positoned in the inial state, and arranges the 3D numpy array of the nest accordingly. This is primarily for visualization purposes since the model does not account for 3D structure.
    
    masksweekobs = masks[c][w+1] # defines the focal nest at the end of the week with w+1
    masksweekobs = arrange_nest(masksweekobs, colonynames[c], c, w, data) # again, arranges the end of week nest according to the positional arrangement of frames from in the inital state 
    
    if downsample: # if downsample is True, reduce the size of the nest arrays by only using every 7th value in the array
        masksweek = masksweek[:,::7,::7] 
        masksweekobs = masksweekobs[:,::7,::7]

    masksweekpred = np.copy(masksweek) # create a predicted 3d nest by copying the initail state nest for now (adding comb will occur below)

    waxbudget = (np.sum(masksweekobs)-np.sum(masksweek))/2 # Calulate the wax budget in comb pixels by subtracting total comb in nest at the end of the week by the comb at the begining of the week (divide by 2 because values for comb in the array are 2, and we just want number of pixelscomb there are)

    while waxbudget>0: # as long as wax budget is not depleted
        f = int(random.random()*10) # pick a random frame
        if np.min(masksweekpred[f])==0: # if this frame is not full of comb
            kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) # create the kernel shape that will be used to dilate the comb. This shape is basically a 3x3 pixel cross with the focal pixel in the center.
            dilation = cv2.dilate(masksweekpred[f],kernel2,iterations = 1) # dilate the comb by 1 iteration (dilation funcion iterates through all the pixels in the 2D array, when it encounters a comb pixel (value of 2 rather than 0) it turns all other pixels in the kernel to comb that are not already)
            masktemp = dilation - masksweekpred[f] # create temporary frame that is only the strip of dilation that occured, this will serve as a template of where random comb can feasably be placed
            masktemp[0,:] = 2 # add a single layer of comb along the the entire top of the frame
            if allsides: # if allsides is True, add a single layer of comb to all sides of frame
                masktemp[-1,:] = 2
                masktemp[:,0] = 2
                masktemp[:,-1] = 2
            maskpotential = masktemp - masksweekpred[f] #remove the comb along the edge where it already exists in the original predicted frame
            maskpotential = np.where(maskpotential==254, 0, maskpotential) # get rid of subtraction artifact values
            potentialpoints = np.argwhere(maskpotential>0) # get the coordinates of all locations that comb can feasably be placed
            randpoint = random.choice(potentialpoints) # choose a random point among these coordinates
            masksweekpred[f][randpoint[0]][randpoint[1]] = 2 # add comb there
            waxbudget = waxbudget - 2 # reduce wax budget


    evaldiff_area = 0 # initialize the variable that will calculate the number of non-overlapping pixels between predicted and observed end-of-week-nests
    obspred_area = 0
    for f in range(len(masksweek)): # for each frame in the nest
        
        maskobs_mask = masksweekobs[f] + masksweek[f] # add together the focal frame at the start of week and observed end of week, resulting in a 2D frame that has double the value of comb where comb is present in both start and end of week frames)
        maskobs_mask = np.where(maskobs_mask == 4, 0, maskobs_mask) # remove comb area present in the initial state (start of week)
        maskobs_mask_area = np.sum(maskobs_mask) # the area of observed new growth

        maskpred_mask = masksweekpred[f] + masksweek[f] # repeat process above but with the predicted end of week growth instead of observed
        maskpred_mask = np.where(maskpred_mask == 4, 0, maskpred_mask)
        maskpred_mask_area = np.sum(maskpred_mask)

        evalim = maskobs_mask + maskpred_mask # add frames of new observed growth and new predicted growth together
        evalim = np.where(evalim == 2, 0, evalim) # wherever there's comb that does NOT overlap, delete it
        evaldiff_area = evaldiff_area + int(np.sum(evalim)) # sum the area where the comb does overlap (needs to be double here to accound for both observed and predicted frames, see below obspred_area_ttl) and add to running total
        obspred_area = obspred_area + (maskobs_mask_area + maskpred_mask_area) # add the total new growth comb area of both observed and predicted (not just overlap but all new comb) to the running nest total new growth
    
    accuracy = (evaldiff_area/obspred_area)*100 # calculate the percentage of the total new oservered nest area that overlaps with the predicted nest area
    areamatch = ((np.sum(masksweekpred)-np.sum(masksweekobs))/np.sum(masksweekobs))*100 # a check to see how close the total predicted area is to the observed area (should be about 100%)
    
    return masksweek, masksweekobs, masksweekpred, accuracy, areamatch