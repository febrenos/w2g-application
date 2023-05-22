import React, { useState } from 'react'
import Header from '../../components/Header/index'

export default function() {
    const [isOpen, setIsOpen] = useState(false);

    return(
        <>
        <main>
            <Header isOpen={isOpen} onClick={() => setIsOpen(!isOpen)}/> 
            <div className={isOpen ? 'contentClose' : 'contentOpen'}>
                <h2>ChatBot</h2>
            </div>
        </main>
        </>
    )
}