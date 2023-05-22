import React, { useState } from 'react'
import HeaderApresentation from '../../components/HeaderApresentation'
import CardApresentation from '../../components/Cards/CardApresentation/index'
import Planet from '../../components/Planet';
import TextApresentation from '../../components/TitleApresentation'
import * as Styled from './style.jsx';

export default function() {

    return(
        <>
            
            <HeaderApresentation/>
            <Styled.ContentTop>
                <TextApresentation/>
                <Planet/>
            </Styled.ContentTop>
            <CardApresentation text="Get Started"/>
        </>
    )
}