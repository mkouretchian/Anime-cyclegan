#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:19:04 2021

@author: roji
"""


import torch
from dataset import PersonAnimeDataset
import sys
from utils import save_checkpoint , load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator



def train_fn(disc_P, disc_A, gen_A, gen_P, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    
    P_reals = 0
    P_fakes = 0
    
    loop = tqdm(loader, leave = True)
    
    for idx , (anime, person) in enumerate(loop):
        anime = anime.to(config.DEVICE)
        person = person.to(config.DEVICE)
        
        
        with torch.cuda.amp.autocast():
            
            fake_person = gen_P(anime)
            D_P_real = disc_P(person)
            D_P_fake = disc_P(fake_person.detach())
            P_reals += D_P_real.mean().item()
            P_fakes += D_P_fake.mean().item()
            D_P_real_loss = mse(D_P_real, torch.ones_like(D_P_real))
            D_P_fake_loss = mse(D_P_fake, torch.zeros_like(D_P_fake))
            D_P_loss = D_P_real_loss + D_P_fake_loss
            
            
            
            
            fake_anime = gen_A(person)
            D_A_real = disc_A(anime)
            D_A_fake = disc_A(fake_anime.detach())
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss
            
            
            D_loss = (D_P_loss + D_A_loss)/2
            
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        
        
        
        with torch.cuda.amp.autocast():
            
            D_P_fake = disc_P(fake_person)
            D_A_fake = disc_A(fake_anime)
            loss_G_P = mse(D_P_fake,torch.ones_like(D_P_fake))
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
            
            
            cycle_anime = gen_A(fake_person)
            cycle_person = gen_P(fake_anime)
            cycle_anime_loss = l1(anime, cycle_anime)
            cycle_person_loss = l1(person, cycle_person)
            
            
            
            
            identity_anime = gen_A(anime)
            identity_person = gen_P(person)
            identity_anime_loss = l1(anime,identity_anime)
            identity_person_loss = l1(person,identity_person)
            
            G_loss = (
                loss_G_A+
                loss_G_P+
                cycle_anime_loss * config.LAMBDA_CYCLE +
                cycle_person_loss * config.LAMBDA_CYCLE +
                identity_anime_loss * config.LAMBDA_IDENTITY+
                identity_person_loss * config.LAMBDA_IDENTITY
                )
            
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        
        
        if idx % 200 == 0 :
            save_image(fake_anime*0.5 + 0.5, f'saved_images/person_{idx}.png')
            save_image(fake_anime*0.5 + 0.5, f'saved_images/anime_{idx}.png')
            
        loop.set_postfix(P_real = P_reals/(idx+1), P_fake = P_fakes/(idx+1))
        
        
        
        
def main():
    
    disc_P = Discriminator(in_channels = 3).to(config.DEVICE)
    disc_A = Discriminator(in_channels = 3).to(config.DEVICE)
    gen_A = Generator(img_channels=3, num_residuals = 9).to(config.DEVICE)
    gen_P = Generator(img_channels=3, num_residuals = 9).to(config.DEVICE)
    
    
    opt_disc = optim.Adam(
        list(disc_P.parameters()) + list(disc_A.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    
    opt_gen = optim.Adam(
        list(gen_A.parameters())+list(gen_P.parameters()),
        lr = config.LEARNING_RATE,
        betas = (0.5,0.999),
        )
    
    
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_P,gen_P,opt_gen, config.LEARNING_RATE,)
        load_checkpoint(
            config.CHECKPOINT_GEN_A, gen_A, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_P, disc_P, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_A, disc_A, opt_disc, config.LEARNING_RATE,
        )
        
    dataset = PersonAnimeDataset(root_anime = config.TRAIN_DIR + "/anime", root_person = config.TRAIN_DIR+"humans",
                                 transform = config.transforms)
    val_dataset = PersonAnimeDataset( root_anime = config.VA"anime", root_person = "cyclegan_test/humans",
                                     transform = config.transforms)
    
    
    val_loader = DataLoader(
        val_dataset,
        batch_size = 1,
        shuffle = False,
        pin_memory = True)
    
    loader = DataLoader(
        dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = True,
        num_workers = config.NUM_WORKERS,
        pin_memory = True)
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_P, disc_A, gen_A, gen_P, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        
        if config.SAVE_MODEL:
            save_checkpoint(gen_P, opt_gen, filename=config.CHECKPOINT_GEN_P)
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_GEN_A)
            save_checkpoint(disc_P, opt_disc, filename=config.CHECKPOINT_CRITIC_P)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_CRITIC_A)

if __name__ == "__main__":
    main()
        
    
    